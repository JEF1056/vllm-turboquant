# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DRY (Don't Repeat Yourself) sampling logits processor."""

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.dry import (
    DRYState,
    _compute_penalties,
)


def _make_state(
    output_token_ids: list[int],
    multiplier: float = 1.0,
    base: float = 1.75,
    allowed_length: int = 2,
    penalty_last_n: int = -1,
    breaker_ids: frozenset[int] | None = None,
) -> DRYState:
    return DRYState(
        multiplier=multiplier,
        base=base,
        allowed_length=allowed_length,
        penalty_last_n=penalty_last_n,
        breaker_ids=breaker_ids or frozenset(),
        output_token_ids=output_token_ids,
    )


class TestComputePenalties:
    """Tests for _compute_penalties — the core DRY matching algorithm."""

    def test_empty_output(self):
        state = _make_state([])
        assert _compute_penalties(state) == {}

    def test_single_token(self):
        state = _make_state([42])
        assert _compute_penalties(state) == {}

    def test_no_repetition(self):
        """All unique tokens → no penalties."""
        state = _make_state([1, 2, 3, 4, 5])
        assert _compute_penalties(state) == {}

    def test_basic_bigram_match(self):
        """Output [A, B, A, B] — suffix [A, B] matches earlier [A, B].
        Token that followed first [A, B] at pos 1 is B (pos 2 is A, pos 3 is B).
        Actually: last token is B (pos 3). Scan for B at earlier positions:
          pos 1: output[1]=B == output[3]=B ✓ → extend backward:
            output[0]=A == output[2]=A ✓ → match_len=2
          match_len=2 >= allowed_length=2 → penalize output[2]=A
          penalty = 1.0 * 1.75^(2-2) = 1.0
        """
        state = _make_state([10, 20, 10, 20], allowed_length=2)
        penalties = _compute_penalties(state)
        assert 10 in penalties
        assert penalties[10] == pytest.approx(1.0)

    def test_longer_match_higher_penalty(self):
        """Output [A, B, C, A, B, C] — suffix [A, B, C] matches earlier.
        Last token C at pos 5. C also at pos 2.
          Extend back: B==B (match_len=2), A==A (match_len=3).
        Penalize output[3]=A with penalty = 1.0 * 1.75^(3-2) = 1.75
        """
        state = _make_state([10, 20, 30, 10, 20, 30], allowed_length=2)
        penalties = _compute_penalties(state)
        assert 10 in penalties
        assert penalties[10] == pytest.approx(1.75)

    def test_exponential_growth(self):
        """Longer matches produce exponentially larger penalties."""
        # [A, B, C, D, A, B, C, D] — match length 4
        tokens = [10, 20, 30, 40, 10, 20, 30, 40]
        state = _make_state(tokens, multiplier=1.0, base=2.0, allowed_length=2)
        penalties = _compute_penalties(state)
        # match_len=4, penalty = 1.0 * 2.0^(4-2) = 4.0
        assert 10 in penalties
        assert penalties[10] == pytest.approx(4.0)

    def test_below_allowed_length_no_penalty(self):
        """Match shorter than allowed_length → no penalty."""
        # [A, B, A, B] with allowed_length=3 → match_len=2 < 3 → no penalty
        state = _make_state([10, 20, 10, 20], allowed_length=3)
        assert _compute_penalties(state) == {}

    def test_allowed_length_one(self):
        """allowed_length=1 penalizes even single-token matches."""
        # [A, B, A] — last token A at pos 2. A at pos 0 → match_len=1.
        # Penalize output[1]=B with penalty = 1.0 * 1.75^(1-1) = 1.0
        state = _make_state([10, 20, 10], allowed_length=1)
        penalties = _compute_penalties(state)
        assert 20 in penalties
        assert penalties[20] == pytest.approx(1.0)

    def test_multiplier_scales_penalty(self):
        """Multiplier linearly scales the penalty."""
        state = _make_state([10, 20, 10, 20], multiplier=0.5, allowed_length=2)
        penalties = _compute_penalties(state)
        assert penalties[10] == pytest.approx(0.5)

    def test_sequence_breaker_interrupts_match(self):
        """A breaker token in the matching sequence stops backward extension."""
        # [A, BREAK, B, A, BREAK, B] — last token B.
        # B at pos 2: extend back → output[1]=BREAK is a breaker → stop.
        # match_len=1 < allowed_length=2 → no penalty.
        BREAK = 99
        state = _make_state(
            [10, BREAK, 20, 10, BREAK, 20],
            allowed_length=2,
            breaker_ids=frozenset([BREAK]),
        )
        assert _compute_penalties(state) == {}

    def test_breaker_as_last_token_still_scanned(self):
        """If the last token itself is a breaker, positions matching it are
        skipped (breaker check on output[i])."""
        BREAK = 99
        # [A, BREAK, A, BREAK] — last token BREAK. Scan for BREAK at pos 1:
        #   output[1] is in breaker_ids → skip (continue).
        state = _make_state(
            [10, BREAK, 10, BREAK],
            allowed_length=1,
            breaker_ids=frozenset([BREAK]),
        )
        assert _compute_penalties(state) == {}

    def test_penalty_last_n_limits_window(self):
        """Only scan the last N tokens."""
        # [A, B, C, D, A, B] — full scan finds match [A, B] at pos 0-1.
        # With penalty_last_n=3, scan_start = 6-3 = 3, only sees [D, A, B].
        # D at pos 3 != B at pos 5 → no match. A at pos 4 != B → no match.
        state = _make_state(
            [10, 20, 30, 40, 10, 20],
            penalty_last_n=3,
            allowed_length=2,
        )
        assert _compute_penalties(state) == {}

    def test_penalty_last_n_finds_match_in_window(self):
        """Match within the scan window is found."""
        # [X, A, B, A, B] — penalty_last_n=4 → scan_start=1
        # Scans positions 1..3. B at pos 2: extend → A==A → match_len=2.
        # Penalize output[3]=A.
        state = _make_state(
            [99, 10, 20, 10, 20],
            penalty_last_n=4,
            allowed_length=2,
        )
        penalties = _compute_penalties(state)
        assert 10 in penalties

    def test_multiple_matches_max_penalty_wins(self):
        """When multiple positions produce penalties for the same next_token,
        the largest penalty should be used."""
        # [A, B, C, A, B, C, A, B, C] — last token C.
        # C at pos 2: extend → B==B, A==A → match_len=3.
        #   Penalize A (pos 3) with 1.0 * 1.75^(3-2) = 1.75
        # C at pos 5: extend → B==B, A==A → match_len=3.
        #   Penalize A (pos 6) with 1.75
        # Both penalize token A with same value.
        state = _make_state(
            [10, 20, 30, 10, 20, 30, 10, 20, 30],
            allowed_length=2,
        )
        penalties = _compute_penalties(state)
        assert 10 in penalties
        assert penalties[10] == pytest.approx(1.75)

    def test_different_tokens_penalized(self):
        """Different next_tokens can each receive penalties."""
        # [A, B, A, C] — last token C != any earlier → no match for C.
        # Actually let's construct: [A, X, A, Y, A, ?]
        # We need the last token to match at multiple positions with
        # different following tokens.
        # [A, X, A, Y, A] — last token A.
        # A at pos 0: match_len=1. If allowed_length=1 → penalize X.
        # A at pos 2: match_len=1. Penalize Y.
        state = _make_state(
            [10, 50, 10, 60, 10],
            allowed_length=1,
        )
        penalties = _compute_penalties(state)
        assert 50 in penalties  # follows first A
        assert 60 in penalties  # follows second A

    def test_disabled_when_multiplier_zero(self):
        """Multiplier 0 means DRY is disabled (returns None from _new_state)."""
        state = _make_state([10, 20, 10, 20], multiplier=0.0)
        # Even if called directly, multiplier=0 produces zero penalties
        penalties = _compute_penalties(state)
        for v in penalties.values():
            assert v == 0.0

    def test_no_match_all_different_last_token(self):
        """Last token doesn't appear earlier → no penalties."""
        state = _make_state([1, 2, 3, 4, 5, 99])
        assert _compute_penalties(state) == {}

    def test_realistic_chat_pattern(self):
        """Simulate a repetitive chat pattern:
        "Hello there Hello there Hello" → tokens [1,2,1,2,1]
        Last token = 1. Match at pos 0 (len=1) and pos 2 (len=2+).
        With allowed_length=2:
          pos 0: output[0]=1==output[4]=1. Extend: nothing before pos 0.
            match_len=1 < 2 → skip.
          pos 2: output[2]=1==output[4]=1. Extend: output[1]=2==output[3]=2.
            match_len=2 >= 2. Extend: output[0]=1==output[2]=1? No, compare
            output[2-2]=output[0]=1 vs output[4-2]=output[2]=1 → yes!
            match_len=3. But wait, i-j=2-2=0 is still >= scan_start=0 and
            n-1-j=4-2=2 >= 0, so continue:
            i-j=2-3=-1 < 0 → stop. match_len=3.
          Penalize output[3]=2 with penalty = 1.0 * 1.75^(3-2) = 1.75
        """
        state = _make_state([1, 2, 1, 2, 1], allowed_length=2)
        penalties = _compute_penalties(state)
        assert 2 in penalties
        assert penalties[2] == pytest.approx(1.75)


class TestDRYStateValidation:
    """Test parameter validation in validate_params."""

    def test_valid_params(self):
        from vllm.v1.sample.logits_processor.dry import DRYLogitsProcessor

        params = SamplingParams(
            extra_args={
                "dry_multiplier": 0.8,
                "dry_base": 1.75,
                "dry_allowed_length": 2,
                "dry_penalty_last_n": -1,
                "dry_sequence_breakers": ["\n", ":"],
            }
        )
        # Should not raise
        DRYLogitsProcessor.validate_params(params)

    def test_no_extra_args(self):
        from vllm.v1.sample.logits_processor.dry import DRYLogitsProcessor

        params = SamplingParams()
        DRYLogitsProcessor.validate_params(params)

    def test_invalid_multiplier(self):
        from vllm.v1.sample.logits_processor.dry import DRYLogitsProcessor

        params = SamplingParams(extra_args={"dry_multiplier": -1.0})
        with pytest.raises(ValueError, match="dry_multiplier"):
            DRYLogitsProcessor.validate_params(params)

    def test_invalid_base(self):
        from vllm.v1.sample.logits_processor.dry import DRYLogitsProcessor

        params = SamplingParams(extra_args={"dry_base": 0})
        with pytest.raises(ValueError, match="dry_base"):
            DRYLogitsProcessor.validate_params(params)

    def test_invalid_allowed_length(self):
        from vllm.v1.sample.logits_processor.dry import DRYLogitsProcessor

        params = SamplingParams(extra_args={"dry_allowed_length": 0})
        with pytest.raises(ValueError, match="dry_allowed_length"):
            DRYLogitsProcessor.validate_params(params)

    def test_invalid_penalty_last_n(self):
        from vllm.v1.sample.logits_processor.dry import DRYLogitsProcessor

        params = SamplingParams(extra_args={"dry_penalty_last_n": 0})
        with pytest.raises(ValueError, match="dry_penalty_last_n"):
            DRYLogitsProcessor.validate_params(params)

    def test_invalid_breakers_type(self):
        from vllm.v1.sample.logits_processor.dry import DRYLogitsProcessor

        params = SamplingParams(extra_args={"dry_sequence_breakers": "not_a_list"})
        with pytest.raises(ValueError, match="dry_sequence_breakers"):
            DRYLogitsProcessor.validate_params(params)


class TestDRYApply:
    """Test the apply() method with mock logits tensors."""

    def test_no_active_requests_passthrough(self):
        """When no requests have DRY enabled, logits are unchanged."""
        from vllm.v1.sample.logits_processor.dry import DRYLogitsProcessor

        proc = DRYLogitsProcessor.__new__(DRYLogitsProcessor)
        proc.req_info = {}

        logits = torch.randn(4, 100)
        original = logits.clone()
        result = proc.apply(logits)
        assert torch.equal(result, original)

    def test_penalties_applied_to_correct_row(self):
        """Penalties only affect the logits row for the DRY-enabled request."""
        from vllm.v1.sample.logits_processor.dry import DRYLogitsProcessor

        proc = DRYLogitsProcessor.__new__(DRYLogitsProcessor)

        # Request at batch index 1 has DRY enabled with a clear repeat
        output_tokens = [10, 20, 10, 20]  # [A, B, A, B]
        proc.req_info = {
            1: _make_state(output_tokens, allowed_length=2),
        }

        logits = torch.zeros(3, 100)
        result = proc.apply(logits)

        # Row 0 (no DRY) should be unchanged
        assert torch.all(result[0] == 0)
        # Row 1, token 10 should be penalized (negative)
        assert result[1, 10] < 0
        # Row 1, other tokens should be unchanged
        assert result[1, 50] == 0
        # Row 2 (no DRY) should be unchanged
        assert torch.all(result[2] == 0)

    def test_penalty_magnitude(self):
        """Verify the penalty value is subtracted from logits."""
        from vllm.v1.sample.logits_processor.dry import DRYLogitsProcessor

        proc = DRYLogitsProcessor.__new__(DRYLogitsProcessor)

        output_tokens = [10, 20, 10, 20]
        proc.req_info = {
            0: _make_state(
                output_tokens,
                multiplier=0.8,
                base=1.75,
                allowed_length=2,
            ),
        }

        logits = torch.zeros(1, 100)
        result = proc.apply(logits)

        expected_penalty = 0.8 * (1.75**0)  # match_len=2, 2-2=0
        assert result[0, 10] == pytest.approx(-expected_penalty, abs=1e-5)
