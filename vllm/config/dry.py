# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for DRY (Don't Repeat Yourself) sampling.

Penalizes tokens based on sequence-level repetition rather than individual
token frequency.  Pass via ``--dry-config`` as a JSON string, e.g.::

    --dry-config '{"multiplier": 0.8, "base": 1.75}'

If ``--dry-config`` is not provided (``None``), DRY sampling is disabled
server-wide.  Per-request overrides via ``extra_args`` still work regardless.
"""

from .utils import config


@config
class DRYConfig:
    """Server-wide defaults for DRY sampling.

    When provided, these values replace the hardcoded defaults.  Per-request
    ``extra_args`` (``dry_multiplier``, ``dry_base``, etc.) still take
    precedence over these server defaults.
    """

    multiplier: float = 0.8
    """Penalty strength.  Higher values penalise repetition more."""

    base: float = 1.75
    """Exponential growth factor for penalty as match length increases."""

    allowed_length: int = 2
    """Minimum match length before a penalty is applied."""

    penalty_last_n: int = -1
    """Context window (in tokens) to scan for repetition.  -1 means all."""

    sequence_breakers: list[str] | None = None
    """Token strings that interrupt repetition matching.
    ``None`` uses the built-in defaults (``["\\n", ":", '"', "*"]``)."""
