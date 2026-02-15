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
from itertools import combinations
from typing import List, Dict, Optional, Tuple

import torch

from .trident import Trident, Droplet


class CoInfo(Trident):
    """Co-information (interaction information) for arbitrary n sources.

    Computes the inclusion-exclusion co-information between n source texts
    and a target text, providing a scalar summary of the redundancy-synergy
    balance without requiring the full PID lattice.

    For n sources X_1, ..., X_n and target Y:

        CI(X_1; ...; X_n; Y) = sum over non-empty S of (-1)^{|S|+1} I(Y; X_S)

    Positive co-information indicates redundancy dominance; negative indicates
    synergy dominance.
    """

    def _compute_subset_mi(
        self,
        posts: List[Droplet],
        target_post_idx: int,
        subset: List[str],
        ll_base: float,
        token_count: int,
    ) -> Optional[float]:
        """Compute I(Y; X_S) for a subset S of source users.

        Returns MI in bits per token, or None if the sequence cannot be built.
        """
        result = self._build_sequence_with_sources(posts, target_post_idx, subset)
        if result is None:
            return None

        tokens, target_span = result
        input_ids = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        ll_subset, _ = self._log_likelihood_simple(input_ids, target_span)

        denom = token_count * math.log(2)
        return (ll_subset - ll_base) / denom

    def compute_pointwise_coinfo(
        self,
        posts: List[Droplet],
        source_users: List[str],
        target_user: str,
        target_post_idx: int,
    ) -> Dict[str, float]:
        """Compute pointwise co-information for a single target post.

        Args:
            posts: List of Droplet posts.
            source_users: User IDs for the n sources (arbitrary length >= 1).
            target_user: User ID for the target.
            target_post_idx: Index of the target post in ``posts``.

        Returns:
            Dictionary with co-information, joint MI, per-source MI values,
            and scored token count. All information values are in bits per token.
        """
        n = len(source_users)

        out = {
            "coinfo_bits_per_token": 0.0,
            "i_y_joint_bits_per_token": 0.0,
            "scored_token_count": 0.0,
        }
        for i, uid in enumerate(source_users):
            out[f"i_y_x{i + 1}_bits_per_token"] = 0.0

        if target_post_idx == 0 or target_post_idx >= len(posts):
            return out
        if posts[target_post_idx].user_id != target_user:
            return out

        # Baseline: target with no sources
        result_base = self._build_sequence_with_sources(posts, target_post_idx, [])
        if result_base is None:
            return out
        tokens_base, target_span_base = result_base
        input_ids_base = torch.tensor(tokens_base, dtype=torch.long, device=self.device).unsqueeze(0)
        ll_base, token_count = self._log_likelihood_simple(input_ids_base, target_span_base)
        if token_count == 0:
            return out

        out["scored_token_count"] = float(token_count)

        # Compute I(Y; X_S) for every non-empty subset S of source_users
        # and accumulate co-information via inclusion-exclusion.
        subset_mis: Dict[tuple, float] = {}
        coinfo = 0.0

        for k in range(1, n + 1):
            sign = (-1) ** (k + 1)  # +1 for singletons, -1 for pairs, +1 for triples, ...
            for combo in combinations(range(n), k):
                subset = [source_users[i] for i in combo]
                mi = self._compute_subset_mi(posts, target_post_idx, subset, ll_base, token_count)
                if mi is None:
                    return out
                subset_mis[combo] = mi
                coinfo += sign * mi

        out["coinfo_bits_per_token"] = float(coinfo)

        # Joint MI: I(Y; X_1, ..., X_n)
        joint_key = tuple(range(n))
        out["i_y_joint_bits_per_token"] = float(subset_mis[joint_key])

        # Per-source MI: I(Y; X_i)
        for i in range(n):
            out[f"i_y_x{i + 1}_bits_per_token"] = float(subset_mis[(i,)])

        return out

    def compute_coinfo(
        self,
        posts: List[Droplet],
        source_users: List[str],
        target_user: str,
    ) -> Tuple[Dict[str, float], List[Dict]]:
        """Compute token-weighted co-information over all target posts.

        Args:
            posts: List of Droplet posts.
            source_users: User IDs for the n sources.
            target_user: User ID for the target.

        Returns:
            (summary, rows) where summary is the token-weighted average and
            rows is a list of per-post pointwise results.
        """
        n = len(source_users)
        rows: List[Dict] = []

        sum_w = 0.0
        sum_coinfo = 0.0
        sum_joint = 0.0
        sum_per_source = [0.0] * n

        for idx, post in enumerate(posts):
            if post.user_id != target_user:
                continue

            point = self.compute_pointwise_coinfo(
                posts=posts,
                source_users=source_users,
                target_user=target_user,
                target_post_idx=idx,
            )

            w = point["scored_token_count"]
            if w <= 0:
                continue

            sum_w += w
            sum_coinfo += w * point["coinfo_bits_per_token"]
            sum_joint += w * point["i_y_joint_bits_per_token"]
            for i in range(n):
                sum_per_source[i] += w * point[f"i_y_x{i + 1}_bits_per_token"]

            rows.append(
                {
                    "source_users": list(source_users),
                    "target_user": target_user,
                    "target_post_id": post.post_id,
                    "target_timestamp": post.timestamp,
                    **point,
                }
            )

        if sum_w == 0.0:
            summary: Dict[str, float] = {
                "coinfo_bits_per_token": 0.0,
                "i_y_joint_bits_per_token": 0.0,
            }
            for i in range(n):
                summary[f"i_y_x{i + 1}_bits_per_token"] = 0.0
            return summary, rows

        summary = {
            "coinfo_bits_per_token": float(sum_coinfo / sum_w),
            "i_y_joint_bits_per_token": float(sum_joint / sum_w),
        }
        for i in range(n):
            summary[f"i_y_x{i + 1}_bits_per_token"] = float(sum_per_source[i] / sum_w)

        return summary, rows
