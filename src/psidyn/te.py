# PSIDyn: Python Semantic Information Dynamics
# Copyright (C) 2025 Leonardo Sebastian Goodall
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
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch

from .trident import Trident, Droplet


class TransferEntropy(Trident):
    def _compute_te_from_tokens(self, input_ids, target_lexical_span, source_spans, return_token_count: bool = False):
        if not source_spans:
            return (0.0, 0) if return_token_count else 0.0

        token_count = target_lexical_span[1] - target_lexical_span[0]
        if token_count <= 0:
            return (0.0, 0) if return_token_count else 0.0

        if not source_spans:
            return (0.0, 0) if return_token_count else 0.0

        ll_full, scored_tokens = self._log_likelihood(
            input_ids,
            target_lexical_span,
            mask_spans=None,
        )
        if scored_tokens == 0:
            return (0.0, 0) if return_token_count else 0.0

        ll_masked, _ = self._log_likelihood(
            input_ids,
            target_lexical_span,
            mask_spans=source_spans,
        )

        te_nats = ll_full - ll_masked
        te_bits = te_nats / (scored_tokens * math.log(2))

        if return_token_count:
            return te_bits, scored_tokens
        return te_bits

    def _compute_pointwise_te(
        self,
        posts,
        source_user,
        target_user,
        target_post_idx,
        lag_window,
        return_token_count: bool = False,
    ):
        if target_post_idx == 0:
            return (0.0, 0) if return_token_count else 0.0

        target_post = posts[target_post_idx]
        if target_post.user_id != target_user:
            return (0.0, 0) if return_token_count else 0.0

        context = self.create_context_and_token_mapping(
            posts, target_post_idx, source_user, lag_window
        )

        if context is None:
            return (0.0, 0) if return_token_count else 0.0

        if not context.source_spans:
            return (0.0, 0) if return_token_count else 0.0

        if context.target_lexical_span[1] <= context.target_lexical_span[0]:
            return (0.0, 0) if return_token_count else 0.0

        input_ids = torch.tensor(
            context.tokens,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)

        return self._compute_te_from_tokens(
            input_ids,
            context.target_lexical_span,
            context.source_spans,
            return_token_count=return_token_count,
        )

    def compute_pointwise_transfer_entropy(self, posts, source_user, target_user, target_post_idx, lag_window):
        return self._compute_pointwise_te(
            posts,
            source_user,
            target_user,
            target_post_idx,
            lag_window,
            return_token_count=False,
        )

    def compute_pointwise_te_with_controls(
        self,
        posts,
        source_user,
        target_user,
        target_post_idx,
        lag_window,
        num_null_samples=0,
        num_placebo_samples=0,
        random_seed=None,
    ):
        """Compute TE along with optional null and placebo controls."""
        te_value = self.compute_pointwise_transfer_entropy(
            posts,
            source_user,
            target_user,
            target_post_idx,
            lag_window,
        )
        results = {
            "te": te_value,
            "null_distribution": [],
            "placebo_distribution": [],
        }

        if num_null_samples <= 0 and num_placebo_samples <= 0:
            return results

        context = self.create_context_and_token_mapping(
            posts, target_post_idx, source_user, lag_window
        )
        if context is None or context.target_lexical_span[1] <= context.target_lexical_span[0]:
            return results

        source_count = len(context.source_spans)
        if source_count == 0:
            return results

        input_ids = torch.tensor(
            context.tokens,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        rng = np.random.default_rng(random_seed)

        if num_null_samples > 0:
            source_span_set = {tuple(span) for span in context.source_spans}
            candidate_spans = [
                span for _, span in context.post_spans
                if tuple(span) not in source_span_set
                and not self._spans_overlap(span, context.target_span)
            ]
            for _ in range(num_null_samples):
                if not candidate_spans:
                    break
                replace = len(candidate_spans) < source_count
                indices = rng.choice(len(candidate_spans), size=source_count, replace=replace)
                sampled_spans = [candidate_spans[idx] for idx in indices]
                te_null = self._compute_te_from_tokens(
                    input_ids,
                    context.target_lexical_span,
                    sampled_spans,
                    return_token_count=False,
                )
                results["null_distribution"].append(te_null)

        if num_placebo_samples > 0 and context.target_history_spans:
            for _ in range(num_placebo_samples):
                replace = len(context.target_history_spans) < source_count
                indices = rng.choice(len(context.target_history_spans), size=source_count, replace=replace)
                sampled_spans = [context.target_history_spans[idx] for idx in indices]
                te_placebo = self._compute_te_from_tokens(
                    input_ids,
                    context.target_lexical_span,
                    sampled_spans,
                    return_token_count=False,
                )
                results["placebo_distribution"].append(te_placebo)

        return results

    def compute_dyadic_transfer_entropy(
        self,
        posts,
        source_user,
        target_user,
        lag_window,
        return_summary: bool = False,
    ):
        """
        Compute dyadic transfer entropy for a source->target pair over all valid target posts.

        Primary estimand:
            Token-weighted mean bits-per-token (i.e. expectation over target tokens).

        TODO: Do we need to return the medians?

        Args:
            posts: Thread of posts
            source_user: Source user Y
            target_user: Target user X
            lag_window: Lag window duration τ
            return_summary: If True, return aggregate TE diagnostics.

        Returns:
            reported_te: token-weighted mean TE in bits per token
            timeseries_rows: list of per-post dictionaries
            metrics_summary (optional): weighted means + medians for diagnostics
        """
        te_values = []
        timeseries_rows = []

        sum_w = 0.0
        sum_te = 0.0

        for i, post in enumerate(posts):
            if post.user_id != target_user:
                continue

            has_source_in_window = any(
                posts[j].user_id == source_user
                and post.timestamp - posts[j].timestamp <= lag_window
                for j in range(i)
            )
            if not has_source_in_window:
                continue

            te_point, token_count = self._compute_pointwise_te(
                posts,
                source_user,
                target_user,
                i,
                lag_window,
                return_token_count=True,
            )

            te_values.append(float(te_point))

            if token_count > 0:
                sum_w += float(token_count)
                sum_te += float(token_count) * float(te_point)

            row = {
                "source_user": source_user,
                "target_user": target_user,
                "target_post_id": post.post_id,
                "target_timestamp": post.timestamp,
                "te_bits_per_token": float(te_point),
                "scored_token_count": float(token_count),
            }
            timeseries_rows.append(row)

        if sum_w > 0.0:
            mean_te = sum_te / sum_w
        else:
            mean_te = 0.0

        reported_te = float(mean_te)

        median_te = float(np.median(te_values)) if te_values else 0.0

        metrics_summary = {
            "mean_te_bits_per_token": float(mean_te),
            "median_te_bits_per_token": float(median_te),
            "total_scored_token_count": float(sum_w),
        }

        if return_summary:
            return reported_te, timeseries_rows, metrics_summary
        return reported_te, timeseries_rows

    def compute_all_dyadic_transfer_entropies(
        self,
        posts,
        lag_window,
        save_te,
        save_timeseries,
        te_csv_path,
        timeseries_csv_path,
        return_summary: bool = False,
    ):
        """
        Compute transfer entropy for all user pairs in the thread.

        Args:
            posts: Thread of posts
            lag_window: Lag window duration τ
            save_te: Whether to save transfer entropies to CSV
            save_timeseries: Whether to save timeseries to CSV
            te_csv_path: Path to save transfer entropies CSV
            timeseries_csv_path: Path to save timeseries CSV
            return_summary: If True, return per-dyad TE summaries.

        Returns:
            results: Dictionary mapping (source_user, target_user) pairs to transfer entropy values
            timeseries_rows: Timeseries rows for all dyads
            summary (optional): Aggregate TE metrics per dyad when requested
        """
        users = list(set(post.user_id for post in posts))
        results: Dict[Tuple[str, str], float] = {}
        all_timeseries: List[Dict[str, Any]] = []
        summary: Dict[Tuple[str, str], Dict[str, float]] = {}

        for source_user in users:
            for target_user in users:
                if source_user == target_user:
                    continue

                if return_summary:
                    te, timeseries_rows, metrics = self.compute_dyadic_transfer_entropy(
                        posts,
                        source_user,
                        target_user,
                        lag_window,
                        return_summary=True,
                    )
                    summary[(source_user, target_user)] = metrics
                else:
                    te, timeseries_rows = self.compute_dyadic_transfer_entropy(
                        posts,
                        source_user,
                        target_user,
                        lag_window,
                        return_summary=False,
                    )

                results[(source_user, target_user)] = te
                all_timeseries.extend(timeseries_rows)

        if save_te:
            output_dir = Path(te_csv_path) if te_csv_path else Path.cwd()
            pd.DataFrame(
                [
                    {"source_user": src, "target_user": tgt, "te_value": value}
                    for (src, tgt), value in results.items()
                ]
            ).to_csv(output_dir / "te.csv", index=False)

        if save_timeseries:
            output_dir = Path(timeseries_csv_path) if timeseries_csv_path else Path.cwd()
            pd.DataFrame(all_timeseries).to_csv(output_dir / "timeseries.csv", index=False)

        if return_summary:
            return results, all_timeseries, summary
        return results, all_timeseries

    def analyse_thread(
        self,
        posts: List[Droplet],
        lag_window: int = 2,
        save_te: bool = False,
        save_timeseries: bool = False,
        te_csv_path: Optional[str] = None,
        timeseries_csv_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of a thread including basic statistics and transfer entropies.

        Args:
            posts: Thread of posts
            lag_window: Lag window duration τ
            save_te: Whether to save transfer entropies to CSV
            save_timeseries: Whether to save timeseries to CSV
            te_csv_path: Path to save transfer entropies CSV
            timeseries_csv_path: Path to save timeseries CSV

        Returns:
            Dictionary with analysis results
        """
        users = list(set(post.user_id for post in posts))
        user_post_counts = defaultdict(int)

        for post in posts:
            user_post_counts[post.user_id] += 1

        transfer_entropies, timeseries_rows = self.compute_all_dyadic_transfer_entropies(
            posts,
            lag_window,
            save_te,
            save_timeseries,
            te_csv_path,
            timeseries_csv_path,
        )

        return {
            "users": users,
            "user_post_counts": dict(user_post_counts),
            "total_posts": len(posts),
            "lag_window": lag_window,
            "transfer_entropies": transfer_entropies,
            "timeseries_rows": timeseries_rows,
        }
