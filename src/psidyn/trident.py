# PSIDyn: Python Semantic Information Dynamics
# Copyright (C) 2026 Leonardo Sebastian Goodall
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None


@dataclass
class Droplet:
    """A linguistic unit: message, post, or other text."""
    user_id: str
    timestamp: int
    content: str = ""
    post_id: str = ""


@dataclass
class ContextMapping:
    """Holds token-level spans needed to realise conditional masking."""
    tokens: List[int]
    source_spans: List[Tuple[int, int]]
    target_span: Tuple[int, int]
    target_lexical_span: Tuple[int, int]
    post_spans: List[Tuple[str, Tuple[int, int]]]
    target_history_spans: List[Tuple[int, int]]


class Trident:
    """
    Estimating semantic information dynamics (semantic transfer entropy; semantic information decomposition).
    """

    def __init__(
        self,
        model_name="meta-llama/Llama-3.2-3B",
        device="cuda" if torch.cuda.is_available() else "cpu",
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
    ):
        """
        Args:
            model_name: HuggingFace model name for the autoregressive transformer
            device: Device to run computations on
            load_in_4bit: If True, load model in 4-bit quantization (default)
            load_in_8bit: If True, load model in 8-bit quantization
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        model_kwargs = {}
        if (load_in_4bit or load_in_8bit) and BitsAndBytesConfig is not None:
            if load_in_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
                )
            else:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            model_kwargs["device_map"] = "auto"

        model_kwargs["attn_implementation"] = "sdpa"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float16),
            low_cpu_mem_usage=True,
            **model_kwargs,
        )

        # If not sharded, move to requested device
        if model_kwargs.get("device_map") is None:
            self.model = self.model.to(device)

        # LLaMA only
        self.model_type = getattr(self.model.config, "model_type", "")
        if self.model_type != "llama":
            raise ValueError(f"Only LLaMA models are supported. Loaded model_type={self.model_type}")

        # Embedding dtype for mask creation
        self._embed_weight_dtype = self.model.model.embed_tokens.weight.dtype

        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        self.attention_mask_fill_value = -1e4

    def create_context_and_token_mapping(self, posts, target_post_idx, source_user, lag_window):
        """Tokenise the thread up to the target post and record spans for masking."""
        target_post = posts[target_post_idx]
        target_time = target_post.timestamp

        sequence_tokens = []
        source_spans = []
        post_spans = []
        target_history_spans = []
        target_span = None
        target_lexical_span = None

        for i in range(target_post_idx + 1):
            post = posts[i]

            prefix_tokens = self.tokenizer.encode(f"[{post.user_id}]: ", add_special_tokens=False)
            content_tokens = self.tokenizer.encode(post.content, add_special_tokens=False)
            newline_tokens = self.tokenizer.encode("\n", add_special_tokens=False)

            span_start = len(sequence_tokens)
            sequence_tokens.extend(prefix_tokens)
            lexical_start = len(sequence_tokens)
            sequence_tokens.extend(content_tokens)
            lexical_end = len(sequence_tokens)
            sequence_tokens.extend(newline_tokens)
            span_end = len(sequence_tokens)

            post_spans.append((post.user_id, (span_start, span_end)))

            if i == target_post_idx:
                target_span = (span_start, span_end)
                target_lexical_span = (lexical_start, lexical_end)
            else:
                time_diff = target_time - post.timestamp
                if post.user_id == source_user and time_diff <= lag_window:
                    source_spans.append((span_start, span_end))
                if post.user_id == target_post.user_id:
                    target_history_spans.append((span_start, span_end))

        if target_span is None or target_lexical_span is None:
            return None

        max_len = self.model.config.max_position_embeddings
        total_len = len(sequence_tokens)
        target_len = target_span[1] - target_span[0]

        if target_len > max_len:
            return None

        if total_len > max_len:
            shift = total_len - max_len
            if shift > target_span[0]:
                return None

            sequence_tokens = sequence_tokens[shift:]

            def adjust_span(span: Tuple[int, int]) -> Optional[Tuple[int, int]]:
                start, end = span
                start -= shift
                end -= shift
                if end <= 0 or start >= len(sequence_tokens):
                    return None
                return max(start, 0), min(end, len(sequence_tokens))

            source_spans = [
                adjusted for span in source_spans
                if (adjusted := adjust_span(span)) is not None and adjusted[1] > adjusted[0]
            ]

            target_history_spans = [
                adjusted for span in target_history_spans
                if (adjusted := adjust_span(span)) is not None and adjusted[1] > adjusted[0]
            ]

            post_spans = [
                (user, adjusted)
                for user, span in post_spans
                if (adjusted := adjust_span(span)) is not None and adjusted[1] > adjusted[0]
            ]

            target_span = adjust_span(target_span)
            target_lexical_span = adjust_span(target_lexical_span)
            if target_span is None or target_lexical_span is None:
                return None

        return ContextMapping(
            tokens=sequence_tokens,
            source_spans=source_spans,
            target_span=target_span,
            target_lexical_span=target_lexical_span,
            post_spans=post_spans,
            target_history_spans=target_history_spans,
        )

    def _build_additive_mask(self, seq_len, custom_blocks, device):
        """Create an additive attention mask applied at every transformer block."""
        # Use in-place operations to reduce memory overhead
        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=self._embed_weight_dtype),
            diagonal=1
        )
        causal.mul_(self.attention_mask_fill_value)
        mask = causal.unsqueeze(0).unsqueeze(0)

        for t0, t1, s0, s1 in custom_blocks:
            t0_clamped = max(0, min(t0, seq_len))
            t1_clamped = max(t0_clamped, min(t1, seq_len))
            s0_clamped = max(0, min(s0, seq_len))
            s1_clamped = max(s0_clamped, min(s1, seq_len))
            if t1_clamped <= t0_clamped or s1_clamped <= s0_clamped:
                continue
            mask[:, :, t0_clamped:t1_clamped, s0_clamped:s1_clamped] += self.attention_mask_fill_value

        return mask

    def _forward_with_mask(self, input_ids, custom_blocks):
        seq_len = input_ids.size(1)
        additive_mask = self._build_additive_mask(seq_len, custom_blocks, device=input_ids.device)

        # LLaMA only: call LlamaModel with additive 4D mask
        llama_model = self.model.model
        position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0)
        outputs = llama_model(
            input_ids=input_ids,
            attention_mask=additive_mask,
            position_ids=position_ids,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
        )
        hidden_states = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.last_hidden_state
        logits = self.model.lm_head(hidden_states)
        return logits

    def _log_likelihood(self, input_ids, target_span, mask_spans=None, cross_mask_spans=None):
        """Return summed log-likelihood over target tokens under an optional mask.

        Args:
            input_ids: Token IDs
            target_span: (start, end) of target tokens to score
            mask_spans: List of (start, end) spans that target cannot attend to
            cross_mask_spans: List of ((observer_start, observer_end), (blocked_start, blocked_end))
                              for masking inter-source attention (e.g., X2 cannot see X1)
        """
        custom_blocks = []
        if mask_spans:
            t0, t1 = target_span
            for s0, s1 in mask_spans:
                if s1 <= s0:
                    continue
                custom_blocks.append((t0, t1, s0, s1))

        # Add cross-source masking blocks (e.g., X2 cannot attend to X1)
        if cross_mask_spans:
            for (obs_start, obs_end), (blk_start, blk_end) in cross_mask_spans:
                if obs_end <= obs_start or blk_end <= blk_start:
                    continue
                custom_blocks.append((obs_start, obs_end, blk_start, blk_end))

        with torch.no_grad():
            logits = self._forward_with_mask(input_ids, custom_blocks=custom_blocks)

        result = self._score_target_tokens(logits, input_ids, target_span)

        # Clean up intermediate tensors
        del logits

        return result

    @staticmethod
    def _spans_overlap(span_a, span_b) -> bool:
        return not (span_a[1] <= span_b[0] or span_b[1] <= span_a[0])

    def _score_target_tokens(self, logits, input_ids, target_span):
        log_probs = torch.log_softmax(logits, dim=-1)

        target_start, target_end = target_span
        if target_start == 0:
            target_start = 1  # Skip first token

        if target_end <= target_start:
            return 0.0, 0

        # Vectorized scoring instead of Python loop
        indices = torch.arange(target_start, target_end, device=input_ids.device)
        token_ids = input_ids[0, target_start:target_end]

        # Get log probs at idx-1 for token at idx
        selected_log_probs = log_probs[0, indices - 1, token_ids]
        total_log_prob = selected_log_probs.sum().item()
        scored_tokens = len(indices)

        return total_log_prob, scored_tokens

    def _get_dual_source_context(self, posts, target_post_idx, source_user_1, source_user_2, lag_window):
        c1 = self.create_context_and_token_mapping(posts, target_post_idx, source_user_1, lag_window)
        if c1 is None:
            return None
        c2 = self.create_context_and_token_mapping(posts, target_post_idx, source_user_2, lag_window)
        if c2 is None:
            return None
        if c1.tokens != c2.tokens or c1.target_lexical_span != c2.target_lexical_span:
            return None

        return {
            "tokens": c1.tokens,
            "target_lexical_span": c1.target_lexical_span,
            "source1_spans": list(c1.source_spans),
            "source2_spans": list(c2.source_spans),
        }

    def _build_sequence_with_sources(
        self,
        posts: List[Droplet],
        target_post_idx: int,
        include_sources: List[str],
    ) -> Optional[Tuple[List[int], Tuple[int, int]]]:
        """
        Build a token sequence including only specified source users.

        No labels/prefixes are added - just raw text separated by newlines.
        This avoids linguistic bias from arbitrary labels like "[premise1]:".

        Args:
            posts: List of Droplet posts
            target_post_idx: Index of the target post
            include_sources: List of user_ids to include (others are omitted)

        Returns:
            (tokens, target_lexical_span) or None if invalid
        """
        sequence_tokens = []
        target_lexical_span = None

        for i in range(target_post_idx + 1):
            post = posts[i]

            # Skip posts from users not in include_sources (unless it's the target)
            if i != target_post_idx and post.user_id not in include_sources:
                continue

            content_tokens = self.tokenizer.encode(post.content, add_special_tokens=False)
            newline_tokens = self.tokenizer.encode("\n", add_special_tokens=False)

            lexical_start = len(sequence_tokens)
            sequence_tokens.extend(content_tokens)
            lexical_end = len(sequence_tokens)
            sequence_tokens.extend(newline_tokens)

            if i == target_post_idx:
                target_lexical_span = (lexical_start, lexical_end)

        if target_lexical_span is None:
            return None

        # Check max length
        max_len = self.model.config.max_position_embeddings
        if len(sequence_tokens) > max_len:
            return None

        return sequence_tokens, target_lexical_span

    def _log_likelihood_simple(self, input_ids: torch.Tensor, target_span: Tuple[int, int]) -> Tuple[float, int]:
        """Compute log-likelihood with standard causal attention (no custom masking)."""
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=False)
            logits = outputs.logits

        return self._score_target_tokens(logits, input_ids, target_span)

    def _compute_two_source_information_terms_mask(
        self,
        input_ids,
        target_lexical_span,
        source1_spans,
        source2_spans,
    ) -> Optional[Tuple[float, float, float, float, float]]:
        """Compute information terms using attention masking (original method)."""
        spans1 = list(source1_spans)
        spans2 = list(source2_spans)

        # Build cross-mask: X2 cannot attend to X1.
        # Sources are independent documents with no natural ordering;
        # the causal position of X2 after X1 is an implementation artifact.
        # Cross-masking ensures sources are mutually invisible, making the
        # computation symmetric (invariant to swapping X1 <-> X2).
        # X1 already cannot attend to X2 under the causal mask.
        source_cross_mask = []
        for s2_span in spans2:
            for s1_span in spans1:
                source_cross_mask.append((s2_span, s1_span))

        # ll_full: target sees both X1 and X2; sources independent of each other
        ll_full, token_count = self._log_likelihood(
            input_ids, target_lexical_span,
            mask_spans=None,
            cross_mask_spans=source_cross_mask if source_cross_mask else None
        )
        if token_count == 0:
            return None

        # ll_base: target sees neither X1 nor X2
        base_spans = spans1 + spans2
        ll_base, _ = self._log_likelihood(
            input_ids, target_lexical_span, mask_spans=base_spans if base_spans else None
        )

        # ll_x1_only: target sees X1, blocked from X2
        # (X1 cannot see X2 under causal mask, so no cross-mask needed)
        ll_x1_only, _ = self._log_likelihood(
            input_ids, target_lexical_span, mask_spans=spans2 if spans2 else None
        )

        # ll_x2_only: target sees X2, blocked from X1; X2 cannot see X1
        ll_x2_only, _ = self._log_likelihood(
            input_ids, target_lexical_span,
            mask_spans=spans1 if spans1 else None,
            cross_mask_spans=source_cross_mask if source_cross_mask else None
        )

        denom = token_count * math.log(2)
        if denom <= 0.0:
            return None

        i1 = (ll_x1_only - ll_base) / denom
        i2 = (ll_x2_only - ll_base) / denom
        i12 = (ll_full - ll_base) / denom

        return float(i1), float(i2), float(i12), float(token_count)

    def _compute_two_source_information_terms_omit(
        self,
        posts: List[Droplet],
        target_post_idx: int,
        source_user_1: str,
        source_user_2: str,
    ) -> Optional[Tuple[float, float, float, float, float]]:
        """
        Compute information terms by physically omitting sources (true marginals).

        Instead of masking attention, builds separate sequences for each condition:
        - I(Y; X1, X2): sequence with both sources
        - I(Y; X1): sequence with only X1
        - I(Y; X2): sequence with only X2
        - I(Y; ∅): sequence with no sources (just target)
        """
        # Build sequence with both sources
        result_both = self._build_sequence_with_sources(
            posts, target_post_idx, [source_user_1, source_user_2]
        )
        if result_both is None:
            return None
        tokens_both, target_span_both = result_both

        input_ids_both = torch.tensor(tokens_both, dtype=torch.long, device=self.device).unsqueeze(0)
        ll_full, token_count = self._log_likelihood_simple(input_ids_both, target_span_both)
        if token_count == 0:
            return None

        # Build sequence with only X1
        result_x1 = self._build_sequence_with_sources(
            posts, target_post_idx, [source_user_1]
        )
        if result_x1 is None:
            return None
        tokens_x1, target_span_x1 = result_x1
        input_ids_x1 = torch.tensor(tokens_x1, dtype=torch.long, device=self.device).unsqueeze(0)
        ll_x1_only, _ = self._log_likelihood_simple(input_ids_x1, target_span_x1)

        # Build sequence with only X2
        result_x2 = self._build_sequence_with_sources(
            posts, target_post_idx, [source_user_2]
        )
        if result_x2 is None:
            return None
        tokens_x2, target_span_x2 = result_x2
        input_ids_x2 = torch.tensor(tokens_x2, dtype=torch.long, device=self.device).unsqueeze(0)
        ll_x2_only, _ = self._log_likelihood_simple(input_ids_x2, target_span_x2)

        # Build sequence with no sources (baseline)
        result_base = self._build_sequence_with_sources(
            posts, target_post_idx, []
        )
        if result_base is None:
            return None
        tokens_base, target_span_base = result_base
        input_ids_base = torch.tensor(tokens_base, dtype=torch.long, device=self.device).unsqueeze(0)
        ll_base, _ = self._log_likelihood_simple(input_ids_base, target_span_base)

        denom = token_count * math.log(2)
        if denom <= 0.0:
            return None

        i1 = (ll_x1_only - ll_base) / denom
        i2 = (ll_x2_only - ll_base) / denom
        i12 = (ll_full - ll_base) / denom

        return float(i1), float(i2), float(i12), float(token_count)

    def compute_post_surprisal(self, posts, target_post_idx, lag_window):
        """
        Compute the average surprisal (bits per token) for a specific post.

        The surprisal is simply the negative log-likelihood of the target tokens
        under the standard causal model conditioned on the preceding context.
        """
        if target_post_idx <= 0 or target_post_idx >= len(posts):
            return None

        target_post = posts[target_post_idx]
        context = self.create_context_and_token_mapping(
            posts,
            target_post_idx,
            source_user=target_post.user_id,
            lag_window=lag_window,
        )
        if context is None or context.target_lexical_span[1] <= context.target_lexical_span[0]:
            return None

        input_ids = torch.tensor(
            context.tokens,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)

        ll_full, token_count = self._log_likelihood(
            input_ids,
            context.target_lexical_span,
            mask_spans=None,
        )
        if token_count == 0:
            return None

        surprisal_bits = -ll_full / (token_count * math.log(2))
        return float(surprisal_bits)

    def compute_surprisal_timeseries(self, posts, lag_window):
        """
        Compute per-post surprisal timeseries for a conversation.
        """
        rows: List[Dict[str, Any]] = []
        for idx, post in enumerate(posts):
            surprisal = self.compute_post_surprisal(posts, idx, lag_window)
            if surprisal is None:
                continue
            rows.append(
                {
                    "target_post_id": post.post_id,
                    "target_user": post.user_id,
                    "target_timestamp": post.timestamp,
                    "surprisal_bits_per_token": surprisal,
                }
            )
        return rows
