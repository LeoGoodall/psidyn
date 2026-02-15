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

from typing import List, Dict, Any, Literal

import torch

from .trident import Trident, Droplet

RedundancyMethod = Literal["mmi", "ccs"]
MarginalizationMethod = Literal["mask", "omit"]


def _sign(x: float, eps: float = 1e-12) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


class PID(Trident):
    def redundancy_mmi(self, i1: float, i2: float, clamp_nonneg: bool = True) -> float:
        r = min(i1, i2)
        return max(0.0, r) if clamp_nonneg else r

    def redundancy_ccs_pointwise(
        self,
        i1: float,
        i2: float,
        i12: float,
        eps: float = 1e-12,
        clamp_nonneg: bool = True,
    ) -> float:
        c = i1 + i2 - i12
        s1, s2, s12, sc = _sign(i1, eps), _sign(i2, eps), _sign(i12, eps), _sign(c, eps)
        keep = (s1 == s2 == s12 == sc) and (sc != 0)
        r = c if keep else 0.0
        return max(0.0, r) if clamp_nonneg else r

    def compute_pointwise_pid(
        self,
        posts: List[Droplet],
        source_user_1: str,
        source_user_2: str,
        target_user: str,
        target_post_idx: int,
        lag_window: int,
        redundancy: RedundancyMethod = "mmi",
        method: MarginalizationMethod = "omit",
        eps: float = 1e-12,
        clamp_nonneg: bool = True,
    ) -> Dict[str, float]:
        """
        Compute pointwise PID for a single target post.

        Args:
            posts: List of Droplet posts
            source_user_1: User ID for source 1
            source_user_2: User ID for source 2
            target_user: User ID for target
            target_post_idx: Index of the target post
            lag_window: Time window for including source posts
            redundancy: Redundancy functional ("mmi" or "ccs")
            method: How to compute marginals:
                - "omit": Physically remove sources from sequence (true marginals)
                - "mask": Use attention masking (sources remain in sequence)
            eps: Epsilon for numerical stability
            clamp_nonneg: Whether to clamp negative values to zero
        """
        out = {
            "i_y_x1_bits_per_token": 0.0,
            "i_y_x2_bits_per_token": 0.0,
            "i_y_x1x2_bits_per_token": 0.0,
            "red_bits_per_token": 0.0,
            "unq_x1_bits_per_token": 0.0,
            "unq_x2_bits_per_token": 0.0,
            "syn_bits_per_token": 0.0,
            "scored_token_count": 0.0,
        }

        if target_post_idx == 0 or target_post_idx >= len(posts):
            return out
        if posts[target_post_idx].user_id != target_user:
            return out

        # Compute information terms using selected method
        if method == "omit":
            info = self._compute_two_source_information_terms_omit(
                posts, target_post_idx, source_user_1, source_user_2
            )
        elif method == "mask":
            ctx = self._get_dual_source_context(posts, target_post_idx, source_user_1, source_user_2, lag_window)
            if ctx is None:
                return out
            t0, t1 = ctx["target_lexical_span"]
            if t1 <= t0:
                return out
            input_ids = torch.tensor(ctx["tokens"], dtype=torch.long, device=self.device).unsqueeze(0)
            info = self._compute_two_source_information_terms_mask(
                input_ids,
                ctx["target_lexical_span"],
                ctx["source1_spans"],
                ctx["source2_spans"],
            )
        else:
            raise ValueError(f"Unknown method='{method}'. Use 'omit' or 'mask'.")

        if info is None:
            return out

        i1, i2, i12, token_count = info
        out["scored_token_count"] = float(token_count)

        if redundancy == "mmi":
            red = self.redundancy_mmi(i1, i2, clamp_nonneg=clamp_nonneg)
        elif redundancy == "ccs":
            red = self.redundancy_ccs_pointwise(i1, i2, i12, eps=eps, clamp_nonneg=clamp_nonneg)
        else:
            raise ValueError(f"Unknown redundancy='{redundancy}'. Use 'mmi' or 'ccs'.")

        unq1 = i1 - red
        unq2 = i2 - red
        syn = i12 - unq1 - unq2 - red

        out.update(
            {
                "i_y_x1_bits_per_token": float(i1),
                "i_y_x2_bits_per_token": float(i2),
                "i_y_x1x2_bits_per_token": float(i12),
                "red_bits_per_token": float(red),
                "unq_x1_bits_per_token": float(unq1),
                "unq_x2_bits_per_token": float(unq2),
                "syn_bits_per_token": float(syn),
            }
        )
        return out

    def compute_pid(
        self,
        posts: List[Droplet],
        source_user_1: str,
        source_user_2: str,
        target_user: str,
        lag_window: int,
        redundancy: RedundancyMethod = "mmi",
        method: MarginalizationMethod = "omit",
        eps: float = 1e-12,
        clamp_nonneg: bool = True,
    ):
        rows = []

        sum_w = 0.0
        sum_i1 = 0.0
        sum_i2 = 0.0
        sum_i12 = 0.0
        sum_red_ccs = 0.0

        for idx, post in enumerate(posts):
            if post.user_id != target_user:
                continue

            point = self.compute_pointwise_pid(
                posts=posts,
                source_user_1=source_user_1,
                source_user_2=source_user_2,
                target_user=target_user,
                target_post_idx=idx,
                lag_window=lag_window,
                redundancy=redundancy,
                method=method,
                eps=eps,
                clamp_nonneg=clamp_nonneg,
            )

            w = point["scored_token_count"]
            if w <= 0:
                continue

            sum_w += w
            sum_i1 += w * point["i_y_x1_bits_per_token"]
            sum_i2 += w * point["i_y_x2_bits_per_token"]
            sum_i12 += w * point["i_y_x1x2_bits_per_token"]

            if redundancy == "ccs":
                sum_red_ccs += w * point["red_bits_per_token"]

            rows.append(
                {
                    "source_user_1": source_user_1,
                    "source_user_2": source_user_2,
                    "target_user": target_user,
                    "target_post_id": post.post_id,
                    "target_timestamp": post.timestamp,
                    **point,
                }
            )

        if sum_w == 0.0:
            summary = {
                "i_y_x1_bits_per_token": 0.0,
                "i_y_x2_bits_per_token": 0.0,
                "i_y_x1x2_bits_per_token": 0.0,
                "red_bits_per_token": 0.0,
                "unq_x1_bits_per_token": 0.0,
                "unq_x2_bits_per_token": 0.0,
                "syn_bits_per_token": 0.0,
                "redundancy_method": redundancy,
            }
            return summary, rows

        avg_i1 = sum_i1 / sum_w
        avg_i2 = sum_i2 / sum_w
        avg_i12 = sum_i12 / sum_w

        if redundancy == "mmi":
            avg_red = self.redundancy_mmi(avg_i1, avg_i2, clamp_nonneg=clamp_nonneg)
        elif redundancy == "ccs":
            avg_red = sum_red_ccs / sum_w
        else:
            raise ValueError(f"Unknown redundancy='{redundancy}'. Use 'mmi' or 'ccs'.")

        avg_unq1 = avg_i1 - avg_red
        avg_unq2 = avg_i2 - avg_red
        avg_syn = avg_i12 - avg_unq1 - avg_unq2 - avg_red

        summary = {
            "i_y_x1_bits_per_token": float(avg_i1),
            "i_y_x2_bits_per_token": float(avg_i2),
            "i_y_x1x2_bits_per_token": float(avg_i12),
            "red_bits_per_token": float(avg_red),
            "unq_x1_bits_per_token": float(avg_unq1),
            "unq_x2_bits_per_token": float(avg_unq2),
            "syn_bits_per_token": float(avg_syn),
            "redundancy_method": redundancy,
        }
        return summary, rows