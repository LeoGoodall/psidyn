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

"""
PSIDyn: Python Semantic Information Dynamics

A Python package for computing information-theoretic measures over natural language
using Large Language Models as semantic probability estimators.

Main components:
- Trident: Base class for semantic information dynamics
- PID: Partial Information Decomposition (redundancy, unique, synergy)
- TransferEntropy: Semantic transfer entropy between text sources
- CoInfo: Co-information (interaction information) for n sources
Example:
    >>> from psidyn import PID, Droplet
    >>> model = PID(model_name="meta-llama/Llama-3.2-3B")
    >>> posts = [
    ...     Droplet(user_id="premise1", timestamp=0, content="The sky is blue"),
    ...     Droplet(user_id="premise2", timestamp=1, content="Blue things are calming"),
    ...     Droplet(user_id="claim", timestamp=2, content="The sky is calming"),
    ... ]
    >>> result = model.compute_pointwise_pid(
    ...     posts, "premise1", "premise2", "claim", target_post_idx=2, lag_window=10
    ... )
"""

__version__ = "0.1.0"

from .trident import Trident, Droplet, ContextMapping
from .pid import PID, RedundancyMethod, MarginalizationMethod
from .te import TransferEntropy
from .coinfo import CoInfo

__all__ = [
    # Version
    "__version__",
    # Core classes
    "Trident",
    "TransferEntropy",
    "PID",
    "CoInfo",
    # Data structures
    "Droplet",
    "ContextMapping",
    # Type aliases
    "RedundancyMethod",
    "MarginalizationMethod",
]
