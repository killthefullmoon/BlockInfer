import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
# Added imports for profiling
import torch
from torch import nn
from contextlib import nullcontext
import torch.profiler as torch_profiler

from blockinfer.config import Config
from blockinfer.sampling_params import SamplingParams
from blockinfer.engine.sequence import Sequence, RunType
from blockinfer.engine.scheduler import Scheduler
from blockinfer.engine.model_runner import ModelRunner
from blockinfer.utils.loader import load_from_hf_model


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        
        self.ps = []
        self.events = []
        
        # Only initialize multi-process tensor parallelism for non-LLaDA models
        # (LLaDA and Dream currently only support single-process execution)
        if config.tensor_parallel_size > 1 and config.model_type not in ("llada", "dream"):
            ctx = mp.get_context("spawn")
            for i in range(1, config.tensor_parallel_size):
                event = ctx.Event()
                process = ctx.Process(target=ModelRunner, args=(config, i, event))
                process.start()
                self.ps.append(process)
                self.events.append(event)
        
        # Initialize model runner
        self.model_runner = ModelRunner(config, 0, self.events)
        
        # Load tokenizer
        print(f"Loading tokenizer from {config.model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True, trust_remote_code=True)
        config.eos = self.tokenizer.eos_token_id
        
        # Handle mask token ID
        if config.mask_token_id is None or config.mask_token_id < 0:
            inferred_mask_id = (
                self.tokenizer.mask_token_id
                if self.tokenizer.mask_token_id is not None
                else self.tokenizer.pad_token_id
            )
            assert inferred_mask_id is not None, "Model tokenizer must have a mask_token_id or pad_token_id"
            config.mask_token_id = inferred_mask_id
        
        print(f"Using mask_token_id: {config.mask_token_id}")

        self.config = config
        self.scheduler = Scheduler(config)
        self.scheduler.consistent_sampling_params = False
        atexit.register(self.exit)

    def offload_parameters(self, include_buffers: bool = False):
        """
        Replace all parameter (and buffer) storages with meta tensors.
        Keeps shapes/dtypes, frees GPU/CPU memory.
        """

        def offload_parameters_keep_buffers(model: torch.nn.Module):
            """
            Move *parameters* to meta to free memory while keeping buffers unchanged.
            Works for any module tree.
            """
            # 1) Snapshot real buffers (module reference + buffer name + tensor)
            saved_buffers = []
            for mod in model.modules():
                for bname, buf in list(mod._buffers.items()):
                    if buf is not None:
                        saved_buffers.append((mod, bname, buf))

            # 2) Move everything to meta
            model.to_empty(device=torch.device("meta"))

            # 3) Restore the saved, real buffers
            for mod, bname, buf in saved_buffers:
                # Reattach the original tensor (device/dtype preserved)
                mod._buffers[bname] = buf

            torch.cuda.empty_cache()
        if include_buffers:
            self.model_runner.model.to_empty(device=torch.device("meta"))
        else:
            offload_parameters_keep_buffers(self.model_runner.model)

        print("Successfully cleaned old parameters (buffers kept)." if not include_buffers
              else "Successfully cleaned old parameters and buffers.")

    def reload_parameters(self, hf_model: nn.Module):
        load_from_hf_model(self.model_runner.model, hf_model=hf_model)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    # ------------------------------------------------------------------
    # Dream diffusion_generate (native) through BlockInfer entrypoint
    # ------------------------------------------------------------------
    def _format_prompt_for_dream(self, prompt: str | list[int]) -> str:
        if isinstance(prompt, list):
            prompt_text = self.tokenizer.decode(prompt, skip_special_tokens=False)
        else:
            prompt_text = prompt
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template:
            prompt_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt_text

    def _dream_sample_tokens(
        self,
        logits: torch.Tensor,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
        margin_confidence: bool = False,
        neg_entropy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if temperature > 0:
            logits = logits / temperature
        if top_p is not None and top_p < 1:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
            mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
        probs = torch.softmax(logits, dim=-1)
        if temperature > 0:
            try:
                x0 = torch.multinomial(probs, num_samples=1).squeeze(-1)
                confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
            except Exception:
                confidence, x0 = probs.max(dim=-1)
        else:
            confidence, x0 = probs.max(dim=-1)
        if margin_confidence:
            sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
            confidence = sorted_probs[:, 0] - sorted_probs[:, 1]
        if neg_entropy:
            eps = 1e-10
            log_probs = torch.log(probs + eps)
            confidence = torch.sum(probs * log_probs, dim=-1)
        return confidence, x0

    def _generate_dream(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = False,
    ) -> list[dict]:
        if not isinstance(prompts, list):
            prompts = [prompts]
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        if any(sp != sampling_params[0] for sp in sampling_params[1:]):
            raise ValueError("Dream generation requires identical sampling_params across the batch.")
        sp = sampling_params[0]

        formatted = [self._format_prompt_for_dream(p) for p in prompts]
        enc = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_model_len,
        )
        input_ids = enc.input_ids.to("cuda")
        attention_mask = enc.attention_mask.to("cuda") if hasattr(enc, "attention_mask") else None

        max_new = sp.max_tokens
        steps = sp.denoising_steps if sp.denoising_steps is not None else max_new
        eps = 1e-3

        batch_size, prompt_len = input_ids.shape
        max_length = prompt_len + max_new
        mask_token_id = self.config.mask_token_id

        # Pad inputs to max_length with mask_token_id
        x = torch.full((batch_size, max_length), mask_token_id, device=input_ids.device, dtype=input_ids.dtype)
        x[:, :prompt_len] = input_ids

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            attn = torch.ones((batch_size, max_length), device=input_ids.device, dtype=attention_mask.dtype)
            attn[:, :prompt_len] = attention_mask
            tok_idx = attn.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attn == 0, 1)
            attention_mask = torch.logical_and(
                attn.unsqueeze(1).unsqueeze(-2),
                attn.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        timesteps = torch.linspace(1, eps, steps + 1, device=input_ids.device)
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True) if use_tqdm else None

        for i in range(steps):
            mask_index = (x == mask_token_id)
            logits = self.model_runner.model(x, attention_mask=attention_mask, tok_idx=tok_idx).logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            mask_logits = logits[mask_index]
            t = timesteps[i]
            s = timesteps[i + 1]

            alg = sp.remasking_strategy
            if alg == "sequential":
                alg_name = "origin"
            elif "entropy" in alg:
                alg_name = "entropy"
            elif "low_confidence" in alg:
                alg_name = "maskgit_plus"
            else:
                alg_name = "origin"

            if alg_name == "origin":
                p_transfer = 1 - s / t if i < steps - 1 else 1
                x0 = torch.full_like(mask_logits, mask_token_id, device=x.device, dtype=torch.long)
                transfer_index_t_s = torch.rand_like(mask_logits, device=x.device) < p_transfer
                if transfer_index_t_s.any():
                    _, sampled = self._dream_sample_tokens(
                        mask_logits[transfer_index_t_s],
                        temperature=sp.temperature,
                        top_p=sp.topp if sp.topp < 1 else None,
                        top_k=sp.topk if sp.topk > 0 else None,
                    )
                    x0[transfer_index_t_s] = sampled
                x[mask_index] = x0.clone()
            else:
                if alg_name == "maskgit_plus":
                    confidence, x0 = self._dream_sample_tokens(
                        mask_logits,
                        temperature=sp.temperature,
                        top_p=sp.topp if sp.topp < 1 else None,
                        top_k=sp.topk if sp.topk > 0 else None,
                    )
                elif alg_name == "entropy":
                    confidence, x0 = self._dream_sample_tokens(
                        mask_logits,
                        temperature=sp.temperature,
                        top_p=sp.topp if sp.topp < 1 else None,
                        top_k=sp.topk if sp.topk > 0 else None,
                        neg_entropy=True,
                    )
                else:
                    raise RuntimeError(f"Unknown alg: {alg_name}")

                num_mask_token = mask_index.sum(dim=-1)
                number_transfer_tokens = (num_mask_token * (1 - s / t)).long()
                number_transfer_tokens = torch.where(number_transfer_tokens > 0, number_transfer_tokens, num_mask_token)

                full_confidence = torch.full_like(x, float("-inf"), device=x.device, dtype=logits.dtype)
                full_confidence[mask_index] = confidence
                x_candidates = torch.full_like(x, mask_token_id, device=x.device, dtype=torch.long)
                x_candidates[mask_index] = x0.clone()

                for b in range(batch_size):
                    k = int(number_transfer_tokens[b].item())
                    if k <= 0:
                        continue
                    fc = full_confidence[b]
                    if sp.dynamic_threshold and sp.dynamic_threshold > 0 and "entropy" in alg_name:
                        fc = fc / sp.dynamic_threshold
                        fc = torch.softmax(fc, dim=-1)
                        indices = torch.multinomial(fc, num_samples=k)
                    else:
                        _, indices = torch.topk(fc, k=k)
                    x[b, indices] = x_candidates[b, indices]

            if pbar:
                pbar.update(0)

        if pbar:
            pbar.close()

        results = []
        for seq in x:
            gen_tokens = seq[prompt_len:].tolist()
            # Align with LLaDA decode: use tokenizer.decode directly
            text = self.tokenizer.decode(gen_tokens)
            results.append({"text": text, "token_ids": gen_tokens})
        return results

    def _format_prompt_for_dream(self, prompt: str | list[int]) -> str:
        if isinstance(prompt, list):
            prompt_text = self.tokenizer.decode(prompt, skip_special_tokens=False)
        else:
            prompt_text = prompt
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template:
            prompt_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt_text

    def _generate_dream(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = False,
    ) -> list[dict]:
        if not isinstance(prompts, list):
            prompts = [prompts]
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        assert len(sampling_params) == len(prompts)
        if any(sp != sampling_params[0] for sp in sampling_params[1:]):
            raise ValueError("Dream generation requires identical sampling_params across the batch.")
        sp = sampling_params[0]

        formatted = [self._format_prompt_for_dream(p) for p in prompts]
        enc = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_model_len,
        )
        input_ids = enc.input_ids.to("cuda")
        attention_mask = enc.attention_mask.to("cuda") if hasattr(enc, "attention_mask") else None

        steps = sp.denoising_steps if sp.denoising_steps is not None else sp.max_tokens
        gen_kwargs = dict(
            max_new_tokens=sp.max_tokens,
            output_history=False,
            return_dict_in_generate=True,
            steps=steps,
            temperature=sp.temperature,
        )
        if sp.topp is not None and sp.topp < 1:
            gen_kwargs["top_p"] = sp.topp
        if sp.topk is not None and sp.topk > 0:
            gen_kwargs["top_k"] = sp.topk
        alg_map = {
            "entropy_bounded": "entropy",
            "low_confidence_static": "maskgit_plus",
            "low_confidence_dynamic": "entropy",
            "low_confidence": "maskgit_plus",
            "sequential": "origin",
        }
        gen_kwargs["alg"] = alg_map.get(sp.remasking_strategy, "entropy")
        gen_kwargs["alg_temp"] = getattr(sp, "dynamic_threshold", 0.0) if "entropy" in gen_kwargs["alg"] else 0.0

        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True) if use_tqdm else None
        with torch.no_grad():
            output = self.model_runner.model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
        if pbar:
            pbar.update(len(prompts))
            pbar.close()

        sequences = output.sequences if hasattr(output, "sequences") else output
        results = []
        eos_token = self.tokenizer.eos_token
        prompt_lens = attention_mask.sum(dim=-1) if attention_mask is not None else torch.tensor(
            [ids.numel() for ids in input_ids], device=input_ids.device
        )
        for prompt_len, seq in zip(prompt_lens.tolist(), sequences):
            generated = seq[prompt_len:].tolist()
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            if eos_token:
                text = text.split(eos_token)[0]
            results.append({"text": text, "token_ids": generated})
        return results

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if isinstance(prompt, list):
            if self.tokenizer.pad_token_id in prompt:
                start = prompt.index(self.tokenizer.pad_token_id) + 1
                prompt = prompt[start:]
        seq = Sequence(prompt, self.config.mask_token_id, sampling_params)
        seq.eos_token_id = self.tokenizer.eos_token_id
        self.scheduler.add(seq)

    def step(self):
        scheduled_seqs, run_type = self.scheduler.schedule()
        if scheduled_seqs is None:
            return [], 0 # Nothing to run

        logits = self.model_runner.call("run", scheduled_seqs, run_type)
        self.scheduler.postprocess(scheduled_seqs, logits, run_type)
        
        finished_outputs = [(seq.seq_id, seq.completion_token_ids) for seq in scheduled_seqs if seq.is_finished]
        
        # Throughput calculation needs to be adapted for block-wise generation
        num_tokens = [self.scheduler.running[i].num_to_transfer if hasattr(self.scheduler.running[i], 'num_to_transfer') else 0 for i in range(len(self.scheduler.running))]
        return finished_outputs, sum(num_tokens)

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        # New optional profiling controls
        profile: bool = False,
        profile_dir: str | None = None,
    ) -> list[str]:
        # Dream uses native diffusion_generate
        if self.config.model_type == "dream":
            return self._generate_dream(prompts, sampling_params, use_tqdm=use_tqdm)
        # ... (This method remains largely the same, but the progress bar will update differently) ...
        # The logic inside the `while not self.is_finished()` loop correctly calls `self.step()`
        # and collects outputs.
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
            self.scheduler.consistent_sampling_params = True
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        
        total_generated_tokens = 0
        start_time = perf_counter()

        # Setup profiler context
        activities = [torch_profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch_profiler.ProfilerActivity.CUDA)
        trace_dir = profile_dir or "profiler_traces"
        prof_ctx = (
            torch_profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                on_trace_ready=torch_profiler.tensorboard_trace_handler(trace_dir),
            )
            if profile else nullcontext()
        )

        with prof_ctx as prof:
            while not self.is_finished():
                output, num_processed = self.step()
                if profile:
                    prof.step()
                total_generated_tokens += num_processed
                
                throughput = total_generated_tokens / (perf_counter() - start_time)
                if use_tqdm:
                    pbar.set_postfix({"Throughput": f"{int(throughput)} tok/s"})

                for seq_id, token_ids in output:
                    outputs[seq_id] = token_ids
                    if use_tqdm:
                        pbar.update(1)

        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs

    def generate_streaming(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        max_active: int | None = None,
        use_tqdm: bool = True,
        # New optional profiling controls
        profile: bool = False,
        profile_dir: str | None = None,
    ) -> list[str]:
        if self.config.model_type == "dream":
            return self._generate_dream(prompts, sampling_params, use_tqdm=use_tqdm)
        """
        Stream prompts through the engine while keeping up to `max_active` sequences running.
        As sequences finish, new prompts are added from the pending list to maximize GPU utilization.
        """
        total = len(prompts)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * total
            self.scheduler.consistent_sampling_params = True

        if max_active is None:
            max_active = getattr(self.scheduler, "max_num_seqs", 32)

        if use_tqdm:
            pbar = tqdm(total=total, desc="Generating", dynamic_ncols=True)

        outputs: dict[int, list[int]] = {}
        pending_idx = 0

        # Prime initial requests up to capacity
        initial = min(max_active, total)
        for i in range(initial):
            self.add_request(prompts[i], sampling_params[i])
        pending_idx = initial

        total_generated_tokens = 0
        start_time = perf_counter()

        # Setup profiler context
        activities = [torch_profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch_profiler.ProfilerActivity.CUDA)
        trace_dir = profile_dir or "profiler_traces"
        prof_ctx = (
            torch_profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                on_trace_ready=torch_profiler.tensorboard_trace_handler(trace_dir),
            )
            if profile else nullcontext()
        )

        with prof_ctx as prof:
            while not self.is_finished() or pending_idx < total:
                # Top up to capacity before each step
                running = getattr(self.scheduler, "running", [])
                deficit = max_active - len(running)
                while deficit > 0 and pending_idx < total:
                    self.add_request(prompts[pending_idx], sampling_params[pending_idx])
                    pending_idx += 1
                    deficit -= 1

                output, num_processed = self.step()
                if profile:
                    prof.step()
                total_generated_tokens += num_processed

                if use_tqdm:
                    throughput = total_generated_tokens / (perf_counter() - start_time + 1e-6)
                    pbar.set_postfix({"Throughput": f"{int(throughput)} tok/s"})
                    pbar.update(len(output))

                for seq_id, token_ids in output:
                    outputs[seq_id] = token_ids

        outputs_list = [outputs[seq_id] for seq_id in sorted(outputs)]
        results = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs_list]

        if use_tqdm:
            pbar.close()
        return results
