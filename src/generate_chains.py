"""
generate_chains.py

Generate multi-step chain-of-thought reasoning traces from a causal LM,
capturing the per-token output distribution at each generation step so
downstream code can compute variance/entropy trajectories.

This module deliberately keeps generation simple (greedy or low-temperature
sampling) so that the variance signal under study comes from the model's
own uncertainty, not from injected randomness.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class StepRecord:
    """One generated token and the distribution it was drawn from."""
    token_id: int
    token_str: str
    probs: torch.Tensor  # full softmax distribution over vocab, shape [vocab_size]


@dataclass
class ChainRecord:
    """A full generated reasoning chain for one problem."""
    prompt: str
    problem_id: str
    generated_text: str
    steps: List[StepRecord] = field(default_factory=list)
    final_answer: Optional[str] = None
    is_correct: Optional[bool] = None


class ChainGenerator:
    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Instruction-tuned models ship a chat template; base models (e.g. plain
        # gpt2) don't. Using the chat template when available is essential —
        # without it, instruct models won't reliably follow the "end with
        # Final Answer: X" instruction, and we saw exactly that failure mode
        # (100% blank extractions) when this was skipped.
        self.has_chat_template = getattr(self.tokenizer, "chat_template", None) is not None

    def _format_prompt(self, prompt: str) -> str:
        if self.has_chat_template:
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt

    @torch.no_grad()
    def generate_chain(
        self,
        prompt: str,
        problem_id: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> ChainRecord:
        """
        Autoregressively generate a completion, recording the full softmax
        distribution at every step. Sampling is temperature/top-p controlled
        but kept modest — the point is to observe naturally-occurring model
        uncertainty, not to inject noise.
        """
        formatted_prompt = self._format_prompt(prompt)
        input_ids = self.tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(self.device)
        generated_ids = input_ids
        record = ChainRecord(prompt=prompt, problem_id=problem_id, generated_text="")

        eos_id = self.tokenizer.eos_token_id

        # KV-cache: feed the full prompt once, then on every subsequent step
        # feed ONLY the single new token and reuse cached key/value states for
        # everything before it. Without this, each step recomputes attention
        # over the entire growing sequence from scratch — O(n^2) total work
        # and the main reason early runs of this script took hours instead of
        # minutes. current_input starts as the full prompt, then collapses to
        # one token per step once past_key_values exists.
        current_input = input_ids
        past_key_values = None

        for _ in range(max_new_tokens):
            outputs = self.model(
                current_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[0, -1, :] / max(temperature, 1e-5)

            # top-p filtering for sampling, but we record the FULL distribution
            # (pre-filtering) for the variance metrics, since we want the
            # model's true uncertainty, not the truncated sampling distribution.
            full_probs = F.softmax(next_token_logits, dim=-1)

            filtered_logits = self._top_p_filter(next_token_logits, top_p)
            sample_probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(sample_probs, num_samples=1)

            token_id = next_token.item()
            token_str = self.tokenizer.decode([token_id])

            record.steps.append(
                StepRecord(token_id=token_id, token_str=token_str, probs=full_probs.cpu())
            )

            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)
            current_input = next_token.unsqueeze(0)  # just the new token next time

            if token_id == eos_id:
                break

        new_tokens = generated_ids[0, input_ids.shape[1]:]
        record.generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return record

    @staticmethod
    def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_mask = cumulative_probs > top_p
        # keep at least one token
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False

        indices_to_remove = sorted_indices[sorted_mask]
        filtered = logits.clone()
        filtered[indices_to_remove] = float("-inf")
        return filtered
