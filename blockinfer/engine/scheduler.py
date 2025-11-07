import torch
from torch.nn import functional as F

from blockinfer.config import Config
from blockinfer.engine.sequence import Sequence, SequenceStatus, RunType
from blockinfer.engine.block_manager import BlockManager
from flashinfer.logits_processor import LogitsPipe, Temperature, Softmax, TopP, TopK


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.mask_token_id = config.mask_token_id
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.running: list[Sequence] = []
        self.sample_pipe = LogitsPipe([
                                Temperature(),      # Scale logits by temperature
                                TopK(),             # Apply top-k filtering
                                Softmax(),          # Convert logits to probabilities
                                TopP(),             # Apply top-p filtering
                            ])
        self.sample_pipe_topk0 = LogitsPipe([
                        Temperature(),      # Scale logits by temperature
                        Softmax(),          # Convert logits to probabilities
                        TopP(),             # Apply top-p filtering
                        ])

    def add(self, seq: Sequence):
        self.running.append(seq)

    def is_finished(self):
        return not self.running

    def schedule(self) -> tuple[list[Sequence], RunType] | tuple[None, None]:
        # 1. Schedule new sequences for prefill
        prefill_candidates = [s for s in self.running if s.status == SequenceStatus.WAITING]
        if prefill_candidates:
            prefill_batch = []
            # Simple batching: take as many as fit
            for seq in prefill_candidates:
                # num_tokens for a waiting seq is its prefill length
                if len(prefill_batch) < self.max_num_seqs and self.block_manager.can_allocate(seq):
                    self.block_manager.allocate(seq)
                    seq.status = SequenceStatus.PREFILLING
                    prefill_batch.append(seq)
            if prefill_batch:
                return prefill_batch, RunType.PREFILL   
        # 2. If no prefilling, create a DENOISE batch.
        denoise_candidates = [s for s in self.running if s.status == SequenceStatus.DENOISING or s.status == SequenceStatus.SAVING]
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
        
        elif run_type == RunType.DENOISE:
            start_idx = 0
            # If all sampling params are the same, compute once and reuse
            if self.consistent_sampling_params:
                if seqs[0].top_k > 0:
                    probs = self.sample_pipe(logits, temperature=seqs[0].temperature, top_k=seqs[0].top_k, top_p=seqs[0].top_p)
                else:
                    probs = self.sample_pipe_topk0(logits, temperature=seqs[0].temperature, top_p=seqs[0].top_p)
                # Single multinomial for the whole batch
                all_sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
                all_sampled_p = torch.gather(probs, -1, all_sampled.unsqueeze(-1)).squeeze(-1)
                # Precompute a device/dtype-consistent -inf scalar
                neg_inf_global = probs.new_full((), float('-inf'))
            for seq in seqs:
                # Extract the part of the tensors relevant to this sequence
                if seq.status == SequenceStatus.DENOISING:
                    block_len = seq.block_length
                    # Cache and reuse the GPU tensor for the intermediate block to avoid re-creation each step
                    t = getattr(seq, "_intermediate_block_tensor", None)
                    if (
                        t is None
                        or t.device != logits.device
                        or t.numel() != block_len
                    ):
                        t = torch.tensor(seq.intermediate_block_tokens, device=logits.device)
                        setattr(seq, "_intermediate_block_tensor", t)
                    current_block_tensor = t
                    mask_index = (current_block_tensor == self.mask_token_id)
                    # If no masks remain, mark as saving/finished without doing sampling work
                    if not mask_index.any():
                        seq.status = SequenceStatus.FINISHED if seq.is_finished else SequenceStatus.SAVING
                        seq.num_to_transfer = 0
                        start_idx += block_len
                        continue

                    if not self.consistent_sampling_params:
                        if seq.top_k > 0:
                            probs = self.sample_pipe(logits[start_idx : start_idx + block_len], temperature=seq.temperature, top_k=seq.top_k, top_p=seq.top_p) 
                        else:
                            probs = self.sample_pipe_topk0(logits[start_idx : start_idx + block_len], temperature=seq.temperature, top_p=seq.top_p)
                        seq_x0 = torch.multinomial(probs, num_samples=1).squeeze(-1) 
                        seq_x0_p = torch.gather(probs, -1, seq_x0.unsqueeze(-1)).squeeze(-1)    
                    else:
                        seq_x0 = all_sampled[start_idx : start_idx + block_len]
                        seq_x0_p = all_sampled_p[start_idx : start_idx + block_len]

                    num_to_transfer = seq.num_transfer_tokens_per_step[seq.current_denoising_step]
                    # Reuse a per-sequence transfer_index buffer to avoid reallocations
                    transfer_index = getattr(seq, "_transfer_index", None)
                    if (
                        transfer_index is None
                        or transfer_index.device != seq_x0.device
                        or transfer_index.numel() != seq_x0.numel()
                    ):
                        transfer_index = torch.zeros_like(seq_x0, dtype=torch.bool)
                        setattr(seq, "_transfer_index", transfer_index)
                    else:
                        transfer_index.zero_()
                    
                    if seq.remasking_strategy == 'sequential':
                        if mask_index.any():
                            first_mask_pos = torch.argmax(mask_index).item()
                            end_pos = min(first_mask_pos + num_to_transfer, block_len)
                            transfer_index[first_mask_pos:end_pos] = True
                    
                    elif 'low_confidence_static' in seq.remasking_strategy:
                        neg_inf = neg_inf_global if self.consistent_sampling_params else seq_x0_p.new_tensor(float('-inf'))
                        confidence = torch.where(mask_index, seq_x0_p, neg_inf)
                        # For dynamic, add threshold logic here if desired
                        _, top_indices = torch.topk(confidence, num_to_transfer)
                        transfer_index[top_indices] = True
                    
                    elif 'low_confidence_dynamic' in seq.remasking_strategy:
                        neg_inf = neg_inf_global if self.consistent_sampling_params else seq_x0_p.new_tensor(float('-inf'))
                        confidence = torch.where(mask_index, seq_x0_p, neg_inf)
                        transfer_index = confidence > seq.dynamic_threshold
                        if int(transfer_index.sum().item()) < num_to_transfer:
                            _, top_indices = torch.topk(confidence, num_to_transfer)
                            transfer_index[top_indices] = True
                        sel = int(transfer_index.sum().item())
                        num_to_transfer = sel if sel > 0 else num_to_transfer
                    elif 'entropy_bounded' in seq.remasking_strategy:
                        block_probs = probs[start_idx : start_idx + block_len]
                        P = block_probs[mask_index]
                        eps = 1e-12
                        entropies = -(P.clamp_min(eps) * (P.clamp_min(eps)).log()).sum(dim=-1)
                        ent_sorted, order = torch.sort(entropies, dim=0, descending=False)
                        cumsum = torch.cumsum(ent_sorted, dim=0)
                        k = torch.searchsorted(cumsum, P.new_tensor(seq.eb_threshold), right=False).item()
                        if k == 0:
                            k = 1
                        # print(k)
                        selected_token_indices = mask_index.nonzero(as_tuple=True)[0][order[:k]]
                        # print(selected_token_indices)
                        transfer_index[selected_token_indices] = True
                        num_to_transfer = k
                    
                    elif seq.remasking_strategy == 'low_confidence':
                        # LLaDA style: transfer tokens with highest confidence (probability)
                        neg_inf = neg_inf_global if self.consistent_sampling_params else seq_x0_p.new_tensor(float('-inf'))
                        confidence = torch.where(mask_index, seq_x0_p, neg_inf)
                        _, top_indices = torch.topk(confidence, min(num_to_transfer, mask_index.sum().item()))
                        transfer_index[top_indices] = True
                    
                    elif seq.remasking_strategy == 'random':
                        # Random remasking: assign random confidence values
                        random_conf = torch.rand(block_len, device=seq_x0.device)
                        neg_inf = neg_inf_global if self.consistent_sampling_params else random_conf.new_tensor(float('-inf'))
                        confidence = torch.where(mask_index, random_conf, neg_inf)
                        _, top_indices = torch.topk(confidence, min(num_to_transfer, mask_index.sum().item()))
                        transfer_index[top_indices] = True

                    # update
                    # In-place update on cached tensor, then sync back to list for IPC compatibility
                    if transfer_index.any():
                        current_block_tensor[transfer_index] = seq_x0[transfer_index]
                    seq.intermediate_block_tokens = current_block_tensor.tolist()
                    
                    seq.current_denoising_step += 1
                    
                    # Check if block is fully denoised
                    is_fully_denoised = (self.mask_token_id not in seq.intermediate_block_tokens) or \
                                        (seq.current_denoising_step >= seq.denoising_steps)

                    if is_fully_denoised:
                        # Block is done, commit it and check if generation is finished
                        seq.status = SequenceStatus.FINISHED if seq.is_finished else SequenceStatus.SAVING
                    seq.num_to_transfer = num_to_transfer
                    
                elif seq.status == SequenceStatus.SAVING:
                    # If saving, commit the block and start a new one
                    seq.commit_block(seq.intermediate_block_tokens)
                    seq.num_to_transfer = 0
                    if not seq.is_finished:
                        seq.start_new_block()

                start_idx += seq.block_length
                
        # Filter out finished sequences from the running list
        finished_seqs = [seq for seq in self.running if seq.is_finished]
        self.running = [seq for seq in self.running if not seq.is_finished]
        for seq in finished_seqs:
            self.block_manager.deallocate(seq)

