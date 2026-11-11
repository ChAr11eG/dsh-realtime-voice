"""Greedy Qwen3-ASR decoding with each token step replayed as one CUDA graph.

Transformers' eager generate() spends ~14 ms per token on kernel launches for this
0.6B model; the graph step takes ~4 ms, halving transcription time with identical text.
Prefill (audio encoder and prompt) stays eager; anything unusual uses the original path.
"""

import threading
from types import SimpleNamespace

import torch
from transformers import StaticCache

# Attention reads the whole static cache each step, so short utterances use a small one.
CACHE_SIZES = (256, 1024)


class GraphDecoder:
    def __init__(self, thinker, size):
        self.thinker = thinker
        device = thinker.device
        self.cache = StaticCache(config=thinker.config.text_config, max_cache_len=size)
        self.token = torch.zeros((1, 1), dtype=torch.long, device=device)
        self.position = torch.zeros((1,), dtype=torch.long, device=device)
        self.delta = torch.zeros((1, 1), dtype=torch.long, device=device)
        self.slots = torch.arange(size, device=device)
        self.mask = torch.zeros((1, 1, 1, size), dtype=torch.bool, device=device)
        self.graph = None

    def _step(self):
        model = self.thinker.model
        # Only slots written by this request are visible; stale entries beyond it are masked.
        self.mask.copy_((self.slots <= self.position).view(1, 1, 1, -1))
        positions = (self.position.view(1, 1) + self.delta).unsqueeze(0).expand(3, 1, 1)
        hidden = model(inputs_embeds=model.embed_tokens(self.token), attention_mask=self.mask,
                       position_ids=positions, past_key_values=self.cache, use_cache=True,
                       cache_position=self.position).last_hidden_state
        return self.thinker.lm_head(hidden[:, -1]).argmax(-1)

    def _capture(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.next_token = self._step()

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, max_new_tokens, eos, **audio):
        length = input_ids.shape[1]
        out = self.thinker(input_ids=input_ids, attention_mask=attention_mask, past_key_values=self.cache,
                           use_cache=True, cache_position=torch.arange(length, device=input_ids.device), **audio)
        token = out.logits[:, -1].argmax(-1)
        self.delta.copy_(self.thinker.rope_deltas.view(1, 1))
        if self.graph is None:
            self.token.copy_(token.view(1, 1))
            self.position.fill_(length)
            self._capture()
        generated = [token]
        for step in range(max_new_tokens - 1):
            if int(token) in eos:
                break
            self.token.copy_(token.view(1, 1))
            self.position.fill_(length + step)
            self.graph.replay()
            token = self.next_token.clone()
            generated.append(token)
        return torch.cat([input_ids, torch.stack(generated, dim=1)], dim=1)


def install():
    """Route single-utterance Qwen3-ASR generation through GraphDecoder."""
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration

    original = Qwen3ASRForConditionalGeneration.generate
    lock = threading.Lock()

    def generate(self, input_ids=None, max_new_tokens=4096, eos_token_id=(151645, 151643), **kwargs):
        mask = kwargs.get("attention_mask")
        needed = input_ids.shape[1] + max_new_tokens if input_ids is not None else None
        size = next((size for size in CACHE_SIZES if needed is not None and needed <= size), None)
        if (size is None or input_ids.shape[0] != 1 or not input_ids.is_cuda
                or (mask is not None and not bool(mask.all()))):
            return original(self, input_ids=input_ids, max_new_tokens=max_new_tokens,
                            eos_token_id=eos_token_id, **kwargs)
        with lock:
            decoders = self.__dict__.setdefault("_graph_decoders", {})
            if size not in decoders:
                decoders[size] = GraphDecoder(self.thinker, size)
            audio = {key: kwargs[key] for key in ("input_features", "feature_attention_mask") if key in kwargs}
            eos = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id)
            return SimpleNamespace(sequences=decoders[size].generate(input_ids, mask, max_new_tokens, eos, **audio))

    Qwen3ASRForConditionalGeneration.generate = generate
