import numpy as np
import torch
from torch.nn import functional as F

from blockinfer.config import Config
from blockinfer.engine.sequence import Sequence, SequenceStatus, RunType
from blockinfer.engine.block_manager import BlockManager
from flashinfer.logits_processor import LogitsPipe, Temperature, Softmax, TopP, TopK


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Add Gumbel noise for sampling, as implemented in Fast-dLLM."""
    if temperature == 0:
        return logits

    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_transfer_index_llada(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    num_transfer_tokens: int,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Determine which tokens should be transferred using LLaDA's approach."""
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == "low_confidence" or "low_confidence" in remasking:
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    elif remasking == "random":
        x0_p = torch.rand_like(x0, dtype=torch.float32)
    else:
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))

    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
    _, select_index = torch.topk(confidence, k=min(num_transfer_tokens, mask_index.sum().item()))
    transfer_index[select_index] = True

    return x0, transfer_index


class Scheduler:
    def __init__(self, config: Config):
        self.config = config
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.pad = getattr(config.hf_config, "pad_token_id", None)
        self.mask_token_id = config.mask_token_id
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.running: list[Sequence] = []
        self.consistent_sampling_params = False

        if config.sampling_method != "gumbel":
            self.sample_pipe = LogitsPipe(
                [
                    Temperature(),
                    TopK(),
                    Softmax(),
                    TopP(),
                ]
            )
            self.sample_pipe_topk0 = LogitsPipe(
                [
                    Temperature(),
                    Softmax(),
                    TopP(),
                ]
            )
        else:
            self.sample_pipe = None
            self.sample_pipe_topk0 = None

    def add(self, seq: Sequence):
        self.running.append(seq)

    def is_finished(self):
        return not self.running

    def schedule(self) -> tuple[list[Sequence], RunType] | tuple[None, None]:
        prefill_candidates = [s for s in self.running if s.status == SequenceStatus.WAITING]
        if prefill_candidates:
            prefill_batch = []
            for seq in prefill_candidates:
                if len(prefill_batch) < self.max_num_seqs and self.block_manager.can_allocate(seq):
                    self.block_manager.allocate(seq)
                    seq.status = SequenceStatus.PREFILLING
                    prefill_batch.append(seq)
            if prefill_batch:
                return prefill_batch, RunType.PREFILL

        denoise_candidates = [s for s in self.running if s.status in (SequenceStatus.DENOISING, SequenceStatus.SAVING)]
        if denoise_candidates:
            denoise_batch = []
            for seq in denoise_candidates:
                num_new_blocks = seq.num_new_blocks_needed(self.block_manager.block_size)
                if len(denoise_batch) < self.max_num_seqs and self.block_manager.can_append_blocks(num_new_blocks):
                    self.block_manager.append_blocks(seq, num_new_blocks)
                    denoise_batch.append(seq)
            if denoise_batch:
                return denoise_batch, RunType.DENOISE

        return None, None

    def postprocess(self, seqs: list[Sequence], logits: torch.Tensor, run_type: RunType):
        if run_type == RunType.PREFILL:
            for seq in seqs:
                seq.num_cached_tokens = seq.num_prefill_tokens
                seq.status = SequenceStatus.DENOISING
            return

        if run_type != RunType.DENOISE:
            return

        if self.config.sampling_method == "gumbel":
            self._postprocess_gumbel(seqs, logits)
            return

        start_idx = 0
        if self.consistent_sampling_params:
            if seqs[0].top_k > 0:
                probs = self.sample_pipe(
                    logits, temperature=seqs[0].temperature, top_k=seqs[0].top_k, top_p=seqs[0].top_p
                )
            else:
                probs = self.sample_pipe_topk0(logits, temperature=seqs[0].temperature, top_p=seqs[0].top_p)
            all_sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
            all_sampled_p = torch.gather(probs, -1, all_sampled.unsqueeze(-1)).squeeze(-1)
            neg_inf_global = probs.new_full((), float("-inf"))

        for seq in seqs:
            if seq.status == SequenceStatus.DENOISING:
                block_len = seq.block_length
                t = getattr(seq, "_intermediate_block_tensor", None)
                if t is None or t.device != logits.device or t.numel() != block_len:
                    t = torch.tensor(seq.intermediate_block_tokens, device=logits.device)
                    setattr(seq, "_intermediate_block_tensor", t)
                current_block_tensor = t
                mask_index = current_block_tensor == self.mask_token_id
                if not mask_index.any():
                    seq.status = SequenceStatus.FINISHED if seq.is_finished else SequenceStatus.SAVING
                    seq.num_to_transfer = 0
                    start_idx += block_len
                    continue

                if not self.consistent_sampling_params:
                    if seq.top_k > 0:
                        probs = self.sample_pipe(
                            logits[start_idx : start_idx + block_len],
                            temperature=seq.temperature,
                            top_k=seq.top_k,
                            top_p=seq.top_p,
                        )
                    else:
                        probs = self.sample_pipe_topk0(
                            logits[start_idx : start_idx + block_len], temperature=seq.temperature, top_p=seq.top_p
                        )
                    seq_x0 = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    seq_x0_p = torch.gather(probs, -1, seq_x0.unsqueeze(-1)).squeeze(-1)
                else:
                    seq_x0 = all_sampled[start_idx : start_idx + block_len]
                    seq_x0_p = all_sampled_p[start_idx : start_idx + block_len]

                num_to_transfer = seq.num_transfer_tokens_per_step[seq.current_denoising_step]
                transfer_index = getattr(seq, "_transfer_index", None)
                if transfer_index is None or transfer_index.device != seq_x0.device or transfer_index.numel() != seq_x0.numel():
                    transfer_index = torch.zeros_like(seq_x0, dtype=torch.bool)
                    setattr(seq, "_transfer_index", transfer_index)
                else:
                    transfer_index.zero_()

                if seq.remasking_strategy == "sequential":
                    if mask_index.any():
                        first_mask_pos = torch.argmax(mask_index).item()
                        end_pos = min(first_mask_pos + num_to_transfer, block_len)
                        transfer_index[first_mask_pos:end_pos] = True

                elif "low_confidence_static" in seq.remasking_strategy:
                    neg_inf = neg_inf_global if self.consistent_sampling_params else seq_x0_p.new_tensor(float("-inf"))
                    confidence = torch.where(mask_index, seq_x0_p, neg_inf)
                    _, top_indices = torch.topk(confidence, num_to_transfer)
                    transfer_index[top_indices] = True

                elif "low_confidence_dynamic" in seq.remasking_strategy:
                    neg_inf = neg_inf_global if self.consistent_sampling_params else seq_x0_p.new_tensor(float("-inf"))
                    confidence = torch.where(mask_index, seq_x0_p, neg_inf)
                    transfer_index = confidence > seq.dynamic_threshold
                    if int(transfer_index.sum().item()) < num_to_transfer:
                        _, top_indices = torch.topk(confidence, num_to_transfer)
                        transfer_index[top_indices] = True
                    sel = int(transfer_index.sum().item())
                    num_to_transfer = sel if sel > 0 else num_to_transfer
                elif "entropy_bounded" in seq.remasking_strategy:
                    block_probs = probs[start_idx : start_idx + block_len]
                    P = block_probs[mask_index]
                    eps = 1e-12
                    entropies = -(P.clamp_min(eps) * (P.clamp_min(eps)).log()).sum(dim=-1)
                    ent_sorted, order = torch.sort(entropies, dim=0, descending=False)
                    cumsum = torch.cumsum(ent_sorted, dim=0)
                    k = torch.searchsorted(cumsum, P.new_tensor(seq.eb_threshold), right=False).item()
                    if k == 0:
                        k = 1
                    selected_token_indices = mask_index.nonzero(as_tuple=True)[0][order[:k]]
                    transfer_index[selected_token_indices] = True
                    num_to_transfer = k

                if transfer_index.any():
                    current_block_tensor[transfer_index] = seq_x0[transfer_index]

                seq.intermediate_block_tokens = current_block_tensor.tolist()
                seq.current_denoising_step += 1

                is_fully_denoised = (self.mask_token_id not in seq.intermediate_block_tokens) or (
                    seq.current_denoising_step >= seq.denoising_steps
                )

                if is_fully_denoised:
                    seq.status = SequenceStatus.FINISHED if seq.is_finished else SequenceStatus.SAVING
                seq.num_to_transfer = num_to_transfer

            elif seq.status == SequenceStatus.SAVING:
                seq.commit_block(seq.intermediate_block_tokens)
                seq.num_to_transfer = 0
                if not seq.is_finished:
                    seq.start_new_block()

            start_idx += seq.block_length

        finished_seqs = [seq for seq in self.running if seq.is_finished]
        self.running = [seq for seq in self.running if not seq.is_finished]
        for seq in finished_seqs:
            self.block_manager.deallocate(seq)

    def _postprocess_gumbel(self, seqs: list[Sequence], logits: torch.Tensor):
        """Postprocess using LLaDA's Gumbel sampling strategy."""
        start_idx = 0

        for seq in seqs:
            if seq.status == SequenceStatus.DENOISING:
                block_len = seq.block_length
                t = getattr(seq, "_intermediate_block_tensor", None)
                if t is None or t.device != logits.device or t.numel() != block_len:
                    t = torch.tensor(seq.intermediate_block_tokens, device=logits.device)
                    setattr(seq, "_intermediate_block_tensor", t)

                current_block_tensor = t
                mask_index = current_block_tensor == self.mask_token_id

                if not mask_index.any():
                    seq.status = SequenceStatus.FINISHED if seq.is_finished else SequenceStatus.SAVING
                    seq.num_to_transfer = 0
                    start_idx += block_len
                    continue

                block_logits = logits[start_idx : start_idx + block_len]
                if getattr(self.config, "model_type", "") == "dream":
                    block_logits = block_logits.clone()
                    if self.eos is not None and 0 <= self.eos < block_logits.size(-1):
                        block_logits[:, self.eos] = float("-inf")
                    if self.pad is not None and self.pad != self.eos and 0 <= self.pad < block_logits.size(-1):
                        block_logits[:, self.pad] = float("-inf")

                num_to_transfer = seq.num_transfer_tokens_per_step[seq.current_denoising_step]

                x0, transfer_index = get_transfer_index_llada(
                    logits=block_logits,
                    temperature=seq.temperature,
                    remasking=seq.remasking_strategy,
                    mask_index=mask_index,
                    x=current_block_tensor,
                    num_transfer_tokens=num_to_transfer,
                    mask_token_id=self.mask_token_id,
                )

                if transfer_index.any():
                    current_block_tensor[transfer_index] = x0[transfer_index]
                seq.intermediate_block_tokens = current_block_tensor.tolist()

                seq.current_denoising_step += 1

                is_fully_denoised = (self.mask_token_id not in seq.intermediate_block_tokens) or (
                    seq.current_denoising_step >= seq.denoising_steps
                )

                if is_fully_denoised:
                    remaining_masks = seq.intermediate_block_tokens.count(self.mask_token_id)
                    print(
                        f"[Debug] Seq {seq.seq_id} block finished after {seq.current_denoising_step} steps; remaining masks={remaining_masks}"
                    )
                    seq.status = SequenceStatus.FINISHED if seq.is_finished else SequenceStatus.SAVING

                seq.num_to_transfer = transfer_index.sum().item()

            elif seq.status == SequenceStatus.SAVING:
                seq.commit_block(seq.intermediate_block_tokens)
                seq.num_to_transfer = 0
                if not seq.is_finished:
                    seq.start_new_block()

            start_idx += seq.block_length

        finished_seqs = [seq for seq in self.running if seq.is_finished]
        self.running = [seq for seq in self.running if not seq.is_finished]
        for seq in finished_seqs:
            self.block_manager.deallocate(seq)
