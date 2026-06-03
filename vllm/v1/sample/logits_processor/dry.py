# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DRY (Don't Repeat Yourself) sampling logits processor.

Penalizes tokens based on sequence-level repetition rather than individual
token frequency. For each candidate token, scans backward through output
to find the longest matching token sequence. Applies an exponentially
growing penalty as match length increases.

Server-wide defaults are configured via ``--dry-config`` JSON, e.g.::

    --dry-config '{"multiplier": 0.8, "base": 1.75}'

If ``--dry-config`` is not provided, DRY sampling is disabled server-wide.
Per-request overrides via ``extra_args`` still work regardless.

Per-request overrides via SamplingParams.extra_args (or vllm_xargs in API):
  dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
  dry_sequence_breakers (list[str])

Precedence: per-request extra_args > DRYConfig server defaults > hardcoded
fallbacks (only used when DRYConfig is None and no per-request override).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Default sequence breakers — tokens that interrupt repetition matching.
_DEFAULT_SEQUENCE_BREAKERS = ["\n", ":", '"', "*"]


@dataclass(frozen=True, slots=True)
class DRYState:
    """Per-request DRY sampling state."""

    multiplier: float
    base: float
    allowed_length: int
    penalty_last_n: int
    breaker_ids: frozenset[int]
    output_token_ids: list[int]  # Reference — always current


# Maximum backward extension depth. Since the penalty exponent is capped
# at _MAX_EXPONENT, extending further produces no additional penalty. This
# turns the worst-case complexity from O(n²) to O(n × _MAX_MATCH_DEPTH).
_MAX_EXPONENT = 50
_MAX_MATCH_DEPTH = _MAX_EXPONENT + 10  # generous headroom above cap


def _compute_penalties(state: DRYState) -> dict[int, float]:
    """Compute DRY penalties for a single request.

    Scans backward through output tokens looking for sequences that match
    the current suffix. For each match, the token that followed the matched
    sequence is penalized.

    Optimizations over naive O(n²):
      1. Match extension depth capped at _MAX_MATCH_DEPTH (penalty is capped
         anyway), turning O(n²) into O(n × constant).
      2. Tokens whose penalty already hit the max are tracked in a set;
         further matches for that token skip the expensive backward scan.

    Returns:
        Mapping of token_id -> penalty value (always positive).
    """
    output = state.output_token_ids
    n = len(output)
    if n < 2:
        return {}

    last_token = output[n - 1]
    breaker_ids = state.breaker_ids

    # Determine scan window
    scan_start = max(0, n - state.penalty_last_n) if state.penalty_last_n > 0 else 0

    penalties: dict[int, float] = {}
    max_depth = state.allowed_length + _MAX_MATCH_DEPTH
    max_penalty = state.multiplier * (state.base**_MAX_EXPONENT)
    # Tokens that already reached maximum penalty — skip further matching.
    saturated: set[int] = set()

    # For each earlier position where the last token appears,
    # extend backward to find the longest matching sequence.
    for i in range(scan_start, n - 1):
        if output[i] != last_token:
            continue
        if output[i] in breaker_ids:
            continue

        # Quick check: if the next_token already has max penalty, skip
        # the expensive backward extension.
        next_token = output[i + 1]
        if next_token in saturated:
            continue

        # Extend backward from position i, capped at max_depth to
        # avoid O(n²) in pathological repeating-pattern cases.
        match_len = 1
        j = 1
        while (i - j) >= scan_start and (n - 1 - j) >= 0 and match_len < max_depth:
            if output[i - j] in breaker_ids:
                break
            if output[i - j] != output[n - 1 - j]:
                break
            match_len += 1
            j += 1

        if match_len < state.allowed_length:
            continue

        exponent = min(match_len - state.allowed_length, _MAX_EXPONENT)
        penalty = state.multiplier * (state.base**exponent)
        prev = penalties.get(next_token, 0.0)
        if penalty > prev:
            penalties[next_token] = penalty
            if penalty >= max_penalty:
                saturated.add(next_token)

    return penalties


def _tokenize_breakers(tokenizer, breakers: list[str]) -> frozenset[int]:
    """Convert breaker strings to token IDs."""
    ids: set[int] = set()
    for breaker in breakers:
        token_ids = tokenizer.encode(breaker, add_special_tokens=False)
        ids.update(token_ids)
    return frozenset(ids)


class DRYLogitsProcessor(LogitsProcessor):
    """DRY (Don't Repeat Yourself) sampling logits processor.

    Penalizes sequence-level repetition with exponentially growing
    penalties. Zero overhead when disabled (multiplier=0).
    """

    @classmethod
    def validate_params(cls, params: SamplingParams):
        ea = params.extra_args
        if not ea:
            return

        multiplier = ea.get("dry_multiplier")
        if multiplier is not None and (
            not isinstance(multiplier, int | float) or multiplier < 0
        ):
            raise ValueError(f"dry_multiplier must be >= 0, got {multiplier}")

        base = ea.get("dry_base")
        if base is not None and (not isinstance(base, int | float) or base <= 0):
            raise ValueError(f"dry_base must be > 0, got {base}")

        allowed_length = ea.get("dry_allowed_length")
        if allowed_length is not None and (
            not isinstance(allowed_length, int) or allowed_length < 1
        ):
            raise ValueError(f"dry_allowed_length must be >= 1, got {allowed_length}")

        penalty_last_n = ea.get("dry_penalty_last_n")
        if penalty_last_n is not None:
            if not isinstance(penalty_last_n, int):
                raise ValueError(
                    f"dry_penalty_last_n must be an int, got {penalty_last_n}"
                )
            if penalty_last_n != -1 and penalty_last_n < 1:
                raise ValueError(
                    f"dry_penalty_last_n must be -1 (all) or >= 1, got {penalty_last_n}"
                )

        breakers = ea.get("dry_sequence_breakers")
        if breakers is not None and (
            not isinstance(breakers, list)
            or not all(isinstance(b, str) for b in breakers)
        ):
            raise ValueError("dry_sequence_breakers must be a list of strings")

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: torch.device,
        is_pin_memory: bool,
    ):
        self.device = device
        self.pin_memory = is_pin_memory
        self.req_info: dict[int, DRYState] = {}

        # Server-wide defaults from DRYConfig (--dry-config JSON).
        # When dry_config is None, DRY is disabled server-wide (multiplier=0).
        # Per-request extra_args can still enable DRY on individual requests.
        dry_cfg = vllm_config.dry_config
        if dry_cfg is not None:
            self._server_multiplier = dry_cfg.multiplier
            self._server_base = dry_cfg.base
            self._server_allowed_length = dry_cfg.allowed_length
            self._server_penalty_last_n = dry_cfg.penalty_last_n
        else:
            # DRY disabled server-wide; per-request overrides still work.
            self._server_multiplier = 0.0
            self._server_base = 1.75
            self._server_allowed_length = 2
            self._server_penalty_last_n = -1

        if self._server_multiplier > 0:
            logger.info(
                "DRY sampling enabled server-wide: multiplier=%.3f, "
                "base=%.2f, allowed_length=%d, penalty_last_n=%d",
                self._server_multiplier,
                self._server_base,
                self._server_allowed_length,
                self._server_penalty_last_n,
            )

        # Pre-compute default breaker token IDs from tokenizer.
        # Use config-provided breakers if set, else built-in defaults.
        breakers = (
            dry_cfg.sequence_breakers
            if dry_cfg is not None and dry_cfg.sequence_breakers is not None
            else _DEFAULT_SEQUENCE_BREAKERS
        )
        self._default_breaker_ids: frozenset[int] = frozenset()
        self._tokenizer = None
        try:
            model_config = vllm_config.model_config
            if model_config and not model_config.skip_tokenizer_init:
                from vllm.tokenizers.registry import get_tokenizer

                tokenizer = get_tokenizer(
                    model_config.tokenizer,
                    trust_remote_code=model_config.trust_remote_code,
                    revision=model_config.tokenizer_revision,
                )
                self._tokenizer = tokenizer
                self._default_breaker_ids = _tokenize_breakers(
                    tokenizer, breakers
                )
        except Exception:
            logger.warning(
                "DRY: Could not load tokenizer for sequence breaker "
                "resolution. Sequence breakers will be disabled unless "
                "dry_sequence_breaker_ids are provided explicitly.",
                exc_info=True,
            )

    def is_argmax_invariant(self) -> bool:
        return False

    def _new_state(
        self,
        params: SamplingParams,
        prompt_tok_ids: list[int] | None,
        output_tok_ids: list[int],
    ) -> DRYState | None:
        ea = params.extra_args or {}
        # Per-request > server config > hardcoded default
        multiplier = ea.get("dry_multiplier", self._server_multiplier)
        if not isinstance(multiplier, int | float) or multiplier <= 0.0:
            return None

        # Resolve breaker IDs: explicit IDs > custom strings > defaults
        breaker_ids_list = ea.get("dry_sequence_breaker_ids")
        if breaker_ids_list is not None:
            breaker_ids = frozenset(int(x) for x in breaker_ids_list)
        elif "dry_sequence_breakers" in ea and self._tokenizer is not None:
            breaker_ids = _tokenize_breakers(
                self._tokenizer, ea["dry_sequence_breakers"]
            )
        else:
            breaker_ids = self._default_breaker_ids

        return DRYState(
            multiplier=float(multiplier),
            base=float(ea.get("dry_base", self._server_base)),
            allowed_length=int(
                ea.get("dry_allowed_length", self._server_allowed_length)
            ),
            penalty_last_n=int(
                ea.get("dry_penalty_last_n", self._server_penalty_last_n)
            ),
            breaker_ids=breaker_ids,
            output_token_ids=output_tok_ids,
        )

    def update_state(self, batch_update: BatchUpdate | None):
        process_dict_updates(
            self.req_info,
            batch_update,
            self._new_state,
        )

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.req_info:
            return logits

        for req_idx, state in self.req_info.items():
            penalties = _compute_penalties(state)
            if not penalties:
                continue

            tok_ids = list(penalties.keys())
            pen_vals = list(penalties.values())

            tok_tensor = torch.tensor(tok_ids, dtype=torch.long, device=logits.device)
            pen_tensor = torch.tensor(
                pen_vals, dtype=logits.dtype, device=logits.device
            )
            logits[req_idx, tok_tensor] -= pen_tensor

        return logits
