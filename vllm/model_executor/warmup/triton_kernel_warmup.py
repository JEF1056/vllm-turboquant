# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Pre-compile Triton JIT kernels that are missed by the standard warmup.

The standard _dummy_run() uses dummy slot mappings (-1) and does not exercise
the speculative decoding preparation/verification loop. This leaves several
Triton kernels uncompiled until the first real inference request, causing
latency spikes flagged by the JIT monitor.

This module provides targeted warmup for four kernel groups:
1. KV cache management (slot mapping, block zeroing)
2. TurboQuant decode attention (stage1 + stage2 + full dequant)
3. Speculative decoding utilities (eagle/MTP prep + rejection sampling)
4. Mamba/FLA hybrid layers (via mixed-batch _dummy_run)

Called from gpu_worker.compile_or_warm_up_model() before JIT monitor
activation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


def _find_attention_group_index(model_runner: GPUModelRunner) -> int:
    """Return the index of the first attention (non-Mamba) KV cache group.

    For hybrid models (e.g., Qwen3.5 with both attention and Mamba layers),
    the first KV cache group may be a Mamba group. Several warmup functions
    need the *attention* group's block table / block_size to match runtime
    constexprs. Using index 0 blindly causes constexpr mismatches on hybrid
    models, leading to JIT recompilation during inference.
    """
    for i, group in enumerate(model_runner.kv_cache_config.kv_cache_groups):
        if isinstance(group.kv_cache_spec, AttentionSpec):
            return i
    return 0  # fallback for non-hybrid models


def warmup_triton_jit_kernels(model_runner: GPUModelRunner) -> None:
    """Pre-compile Triton JIT kernels not covered by standard warmup.

    Each sub-function is guarded by feature detection so only relevant
    kernels are warmed up. All dummy tensors are allocated, used for
    compilation, then freed.
    """
    device = model_runner.device

    warmup_kv_cache_kernels(model_runner, device)

    if _has_turboquant_backend(model_runner):
        warmup_turboquant_decode_kernels(model_runner, device)

    if model_runner.speculative_config is not None:
        warmup_spec_decode_kernels(model_runner, device)

    if model_runner.model_config.is_hybrid:
        warmup_mamba_hybrid_kernels(model_runner)
        warmup_batch_memcpy_kernel(device)

        # The postprocess kernel is only used when spec decode + hybrid.
        if model_runner.speculative_config is not None:
            warmup_postprocess_mamba_kernel(model_runner, device)

    torch.cuda.synchronize(device)
    logger.info("Triton JIT kernel warmup completed.")


# ---------------------------------------------------------------------------
# Group 1: KV cache management kernels
# ---------------------------------------------------------------------------


def warmup_kv_cache_kernels(model_runner: GPUModelRunner, device: torch.device) -> None:
    """Warm up _zero_kv_blocks_kernel and _compute_slot_mapping_kernel."""
    if not HAS_TRITON:
        return
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID
    from vllm.v1.worker.block_table import _compute_slot_mapping_kernel
    from vllm.v1.worker.utils import _zero_kv_blocks_kernel

    # --- _zero_kv_blocks_kernel ---
    # Use the existing KVBlockZeroer if it's been initialized; otherwise
    # do a minimal direct kernel launch with representative constexprs.
    block_zeroer = model_runner._kv_block_zeroer
    if block_zeroer._meta is not None:
        # Trigger a zero on a single dummy block (block 0 = null block).
        block_zeroer.zero_block_ids([0])
        logger.debug("Warmed up _zero_kv_blocks_kernel via KVBlockZeroer.")
    else:
        # Fallback: direct launch with small representative shapes.
        n_segs = 2
        page_size_el = 1024
        blk_size = 256
        seg_addrs = torch.zeros(n_segs, dtype=torch.uint64, device=device)
        # Allocate a small scratch buffer and use its address.
        scratch = torch.zeros(n_segs * page_size_el, dtype=torch.int32, device=device)
        for i in range(n_segs):
            seg_addrs[i] = scratch.data_ptr() + i * page_size_el * 4
        block_ids = torch.zeros(1, dtype=torch.int64, device=device)
        grid = (1 * n_segs * (page_size_el // blk_size),)
        _zero_kv_blocks_kernel[grid](
            seg_addrs,
            block_ids,
            1,
            N_SEGS=n_segs,
            PAGE_SIZE_EL=page_size_el,
            BLOCK_SIZE=blk_size,
        )
        del scratch
        logger.debug("Warmed up _zero_kv_blocks_kernel via direct launch.")

    # --- _compute_slot_mapping_kernel ---
    num_reqs = 2
    num_tokens = 4
    max_num_tokens = max(
        num_tokens, model_runner.scheduler_config.max_num_batched_tokens
    )
    max_num_blocks_per_req = 4
    block_size = 16
    # Read the real block size and cp_kv_cache_interleave_size from the
    # attention group's block table to match runtime constexprs exactly.
    # For hybrid models (e.g., Qwen3.5), group 0 may be a Mamba group
    # whose block_size differs from the attention block_size.
    attn_group_idx = _find_attention_group_index(model_runner)
    bt = model_runner.input_batch.block_table.block_tables[attn_group_idx]
    block_size = bt.block_size
    cp_kv_cache_interleave_size = bt.cp_kv_cache_interleave_size
    max_num_blocks_per_req = bt.max_num_blocks_per_req

    query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32, device=device)
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    block_table = torch.ones(
        num_reqs, max_num_blocks_per_req, dtype=torch.int32, device=device
    )
    slot_mapping = torch.full(
        (max_num_tokens,), PAD_SLOT_ID, dtype=torch.int64, device=device
    )

    _compute_slot_mapping_kernel[(num_reqs + 1,)](
        num_tokens,
        max_num_tokens,
        query_start_loc,
        positions,
        block_table,
        block_table.stride(0),
        block_size,
        slot_mapping,
        TOTAL_CP_WORLD_SIZE=1,
        TOTAL_CP_RANK=0,
        CP_KV_CACHE_INTERLEAVE_SIZE=cp_kv_cache_interleave_size,
        PAD_ID=PAD_SLOT_ID,
        BLOCK_SIZE=1024,
    )
    logger.debug("Warmed up _compute_slot_mapping_kernel.")


# ---------------------------------------------------------------------------
# Group 2: TurboQuant decode attention kernels
# ---------------------------------------------------------------------------


def _has_turboquant_backend(model_runner: GPUModelRunner) -> bool:
    cache_dtype = model_runner.cache_config.cache_dtype
    return cache_dtype.startswith("turboquant_")


def warmup_turboquant_decode_kernels(
    model_runner: GPUModelRunner, device: torch.device
) -> None:
    """Warm up _tq_decode_stage1, _fwd_kernel_stage2, _tq_full_dequant_kv."""
    from vllm.model_executor.layers.quantization.turboquant.config import (
        TurboQuantConfig,
    )
    from vllm.v1.attention.ops.triton_turboquant_decode import (
        triton_turboquant_decode_attention,
    )

    cache_dtype = model_runner.cache_config.cache_dtype
    head_size = model_runner.model_config.get_head_size()
    tq_config = TurboQuantConfig.from_cache_dtype(cache_dtype, head_size)

    num_kv_heads = model_runner.model_config.get_num_kv_heads(
        model_runner.parallel_config
    )
    num_attention_heads = model_runner.model_config.get_num_attention_heads(
        model_runner.parallel_config
    )
    D = head_size
    Hq = num_attention_heads
    Hk = num_kv_heads

    # Build minimal dummy tensors.
    BQ = 2  # 2 query positions
    # Use the attention group's block_size — for hybrid models, group 0
    # may be a Mamba group with a different block_size.
    attn_group_idx = _find_attention_group_index(model_runner)
    kv_groups = model_runner.kv_cache_config.kv_cache_groups
    block_size = kv_groups[attn_group_idx].kv_cache_spec.block_size if kv_groups else 16
    num_blocks = 2
    slot_size_aligned = tq_config.slot_size_aligned
    padded_slot = slot_size_aligned // 2  # stored as int16 -> byte pairs

    query = torch.randn(BQ, Hq, D, dtype=torch.bfloat16, device=device)
    kv_cache = torch.zeros(
        num_blocks,
        block_size,
        Hk,
        padded_slot,
        dtype=torch.uint8,
        device=device,
    )
    block_table = torch.zeros(BQ, num_blocks, dtype=torch.int32, device=device)
    # Each request has seq_len >= 1 so stage1 has work to do.
    seq_lens = torch.ones(BQ, dtype=torch.int32, device=device)
    Pi = torch.eye(D, dtype=torch.float32, device=device)
    centroids = torch.linspace(
        -1.0,
        1.0,
        tq_config.n_centroids,
        dtype=torch.float32,
        device=device,
    )
    scale = 1.0 / math.sqrt(D)

    triton_turboquant_decode_attention(
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        Pi=Pi,
        centroids=centroids,
        scale=scale,
        mse_bits=tq_config.mse_bits,
        key_packed_size=tq_config.key_packed_size,
        value_quant_bits=tq_config.effective_value_quant_bits,
        key_fp8=tq_config.key_fp8,
        norm_correction=tq_config.norm_correction,
        query_len=1,
    )
    logger.debug("Warmed up TQ decode kernels (_tq_decode_stage1, _fwd_kernel_stage2).")

    # Also warm up _tq_full_dequant_kv for continuation prefill path.
    _warmup_tq_full_dequant(tq_config, kv_cache, block_table, centroids, Hk, D, device)


def _warmup_tq_full_dequant(
    tq_config,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    centroids: torch.Tensor,
    Hk: int,
    D: int,
    device: torch.device,
) -> None:
    """Warm up _tq_full_dequant_kv kernel."""
    import triton

    from vllm.v1.attention.ops.triton_turboquant_decode import (
        _tq_full_dequant_kv,
        _use_fp8_e4b15,
    )

    B = block_table.shape[0]
    block_size = kv_cache.shape[1]
    mse_bits = tq_config.mse_bits
    mse_bytes = math.ceil(D * mse_bits / 8)
    val_data_bytes = math.ceil(D * tq_config.effective_value_quant_bits / 8)
    BLOCK_D = triton.next_power_of_2(D)

    # Output buffers for dequantized K and V.
    K_out = torch.empty(B, Hk, block_size, D, dtype=torch.float16, device=device)
    V_out = torch.empty(B, Hk, block_size, D, dtype=torch.float16, device=device)

    fp8_e4b15 = _use_fp8_e4b15(device.index or 0)

    grid = (B, Hk)
    _tq_full_dequant_kv[grid](
        kv_cache,
        block_table,
        centroids,
        K_out,
        V_out,
        K_out.stride(0),
        K_out.stride(1),
        K_out.stride(2),
        V_out.stride(0),
        V_out.stride(1),
        V_out.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        HEAD_DIM=D,
        BLOCK_SIZE=block_size,
        NUM_KV_HEADS=Hk,
        MSE_BYTES=mse_bytes,
        KPS=tq_config.key_packed_size,
        VQB=tq_config.effective_value_quant_bits,
        VAL_DATA_BYTES=val_data_bytes,
        MSE_BITS=mse_bits,
        KEY_FP8=1 if tq_config.key_fp8 else 0,
        BLOCK_D=BLOCK_D,
        NORM_CORRECTION=1 if tq_config.norm_correction else 0,
        FP8_E4B15=fp8_e4b15,
    )
    logger.debug("Warmed up _tq_full_dequant_kv.")


# ---------------------------------------------------------------------------
# Group 3: Speculative decoding kernels
# ---------------------------------------------------------------------------


def warmup_spec_decode_kernels(
    model_runner: GPUModelRunner, device: torch.device
) -> None:
    """Warm up eagle/MTP preparation and rejection sampling kernels."""
    if not HAS_TRITON:
        return
    from vllm.v1.sample.rejection_sampler import (
        MAX_SPEC_LEN,
        expand_kernel,
    )
    from vllm.v1.spec_decode.utils import (
        eagle_prepare_inputs_padded_kernel,
        eagle_prepare_next_token_padded_kernel,
        eagle_step_slot_mapping_metadata_kernel,
    )

    num_reqs = 2
    num_spec_tokens = model_runner.speculative_config.num_speculative_tokens
    vocab_size = model_runner.model_config.get_vocab_size()
    # Read block_size and n_blocks_per_req from the attention group's
    # block table to match runtime constexprs exactly. For hybrid models
    # (e.g., Qwen3.5), group 0 may be a Mamba group whose block_size
    # differs from the attention block_size used by spec decode.
    attn_group_idx = _find_attention_group_index(model_runner)
    bt = model_runner.input_batch.block_table.block_tables[attn_group_idx]
    block_size = bt.block_size
    max_model_len = model_runner.model_config.max_model_len
    n_blocks_per_req = bt.max_num_blocks_per_req

    # --- eagle_step_slot_mapping_metadata_kernel ---
    positions = torch.zeros(num_reqs, dtype=torch.int64, device=device)
    block_table = torch.ones(
        num_reqs, n_blocks_per_req, dtype=torch.int32, device=device
    )
    seq_lens = torch.ones(num_reqs, dtype=torch.int32, device=device)
    out_positions = torch.empty(num_reqs, dtype=torch.int64, device=device)
    out_slot_mapping = torch.empty(num_reqs, dtype=torch.int64, device=device)

    eagle_step_slot_mapping_metadata_kernel[(num_reqs,)](
        positions,
        block_table,
        block_table.stride(0),
        seq_lens,
        out_positions,
        out_slot_mapping,
        block_size=block_size,
        max_model_len=max_model_len,
        n_blocks_per_req=n_blocks_per_req,
        PAD_ID=-1,
        batch_size=num_reqs,
    )
    logger.debug("Warmed up eagle_step_slot_mapping_metadata_kernel.")

    # --- eagle_prepare_next_token_padded_kernel ---
    num_sampled_per_req = num_spec_tokens + 1
    from vllm.v1.spec_decode.utils import next_power_of_2

    BLOCK_SIZE_TOKENS = next_power_of_2(num_sampled_per_req)

    # Use int32 dtypes to match the actual rejection sampler output
    # (output_token_ids is int32) and the runtime int32 buffers for
    # backup_tokens, next_token_ids, valid_sampled_tokens_count.
    sampled_token_ids = torch.full(
        (num_reqs, num_sampled_per_req),
        -1,
        dtype=torch.int32,
        device=device,
    )
    discard_mask = torch.zeros(num_reqs, dtype=torch.bool, device=device)
    backup_tokens = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    next_token_ids = torch.empty(num_reqs, dtype=torch.int32, device=device)
    valid_counts = torch.empty(num_reqs, dtype=torch.int32, device=device)

    eagle_prepare_next_token_padded_kernel[(num_reqs,)](
        sampled_token_ids,
        discard_mask,
        backup_tokens,
        next_token_ids,
        valid_counts,
        vocab_size,
        num_sampled_per_req,
        num_reqs,
        sampled_token_ids.stride(0),
        BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
    )
    logger.debug("Warmed up eagle_prepare_next_token_padded_kernel.")

    # --- eagle_prepare_inputs_padded_kernel ---
    cu_num_draft = torch.arange(1, num_reqs + 1, dtype=torch.int32, device=device)
    # Use int32 to match runtime: valid_sampled_tokens_count is created
    # via next_token_ids.new_empty() where next_token_ids is int32.
    valid_sampled_counts = torch.ones(num_reqs, dtype=torch.int32, device=device)
    query_start_loc = torch.arange(0, num_reqs + 1, dtype=torch.int32, device=device)
    token_indices = torch.empty(num_reqs, dtype=torch.int32, device=device)
    num_rejected = torch.empty(num_reqs, dtype=torch.int32, device=device)

    eagle_prepare_inputs_padded_kernel[(num_reqs,)](
        cu_num_draft,
        valid_sampled_counts,
        query_start_loc,
        token_indices,
        num_rejected,
        num_reqs,
    )
    logger.debug("Warmed up eagle_prepare_inputs_padded_kernel.")

    # --- expand_kernel ---
    # Runtime creates cu_num_draft_tokens via np.cumsum(..., dtype=np.int32)
    # (see spec_decode/metadata.py), so warmup must use int32 to avoid a
    # Triton type mismatch between the literal 0 (int32) in the then-branch
    # and tl.load (int64) in the else-branch.
    batch_size = num_reqs
    total_tokens = batch_size * 2
    x = torch.zeros(batch_size, dtype=torch.int32, device=device)
    cu_num_tokens = torch.arange(
        2,
        2 * batch_size + 1,
        step=2,
        dtype=torch.int32,
        device=device,
    )
    expanded = torch.empty(total_tokens, dtype=torch.int32, device=device)

    expand_kernel[(batch_size,)](
        expanded,
        x,
        cu_num_tokens,
        0,
        0,
        MAX_NUM_TOKENS=MAX_SPEC_LEN,
    )
    logger.debug("Warmed up expand_kernel.")


# ---------------------------------------------------------------------------
# Group 4: Mamba/FLA hybrid model kernels
# ---------------------------------------------------------------------------


def warmup_mamba_hybrid_kernels(model_runner: GPUModelRunner) -> None:
    """Warm up Mamba causal_conv1d and FLA/GDN kernels.

    Runs _dummy_run with create_mixed_batch=True to exercise both prefill
    and decode code paths through Mamba/FLA layers. Also runs a uniform
    decode dummy to cover the causal_conv1d_update kernel specifically.
    """
    from vllm.config.compilation import CUDAGraphMode

    max_num_seqs = model_runner.scheduler_config.max_num_seqs
    # Mixed batch: decode tokens + one prefill request.
    # Need at least max_num_seqs + 1 tokens for a meaningful mixed batch.
    mixed_tokens = max(16, max_num_seqs + 1)
    mixed_tokens = min(mixed_tokens, model_runner.max_num_tokens)

    model_runner._dummy_run(
        num_tokens=mixed_tokens,
        skip_eplb=True,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        force_attention=True,
        create_mixed_batch=True,
    )
    logger.debug("Warmed up Mamba/FLA prefill+decode paths via mixed-batch dummy run.")

    # Uniform decode run to cover causal_conv1d_update_kernel specifically.
    decode_tokens = min(max_num_seqs, model_runner.max_num_tokens)
    model_runner._dummy_run(
        num_tokens=decode_tokens,
        skip_eplb=True,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        force_attention=True,
        uniform_decode=True,
    )
    logger.debug("Warmed up Mamba/FLA decode path via uniform-decode dummy run.")


# ---------------------------------------------------------------------------
# Group 4b: batch_memcpy_kernel (Mamba state copy during prefix caching)
# ---------------------------------------------------------------------------


def warmup_batch_memcpy_kernel(device: torch.device) -> None:
    """Warm up batch_memcpy_kernel used for Mamba state block copies.

    This kernel is triggered by do_mamba_copy_block() when prefix caching
    requires copying Mamba states between blocks. The _dummy_run() warmup
    does not exercise this path, so it must be compiled explicitly.
    """
    if not HAS_TRITON:
        return
    from vllm.v1.worker.mamba_utils import batch_memcpy_kernel

    # Minimal launch: 1 copy operation with BLOCK_SIZE=1024 (matches runtime).
    n = 1
    scratch = torch.zeros(2048, dtype=torch.uint8, device=device)
    src_ptrs = torch.tensor(
        [scratch.data_ptr()], dtype=torch.int64, device=device
    )
    dst_ptrs = torch.tensor(
        [scratch.data_ptr() + 1024], dtype=torch.int64, device=device
    )
    sizes = torch.tensor([512], dtype=torch.int32, device=device)

    batch_memcpy_kernel[(n,)](src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=1024)
    del scratch
    logger.debug("Warmed up batch_memcpy_kernel.")


# ---------------------------------------------------------------------------
# Group 5: Mamba postprocess kernel (spec decode + hybrid only)
# ---------------------------------------------------------------------------


def warmup_postprocess_mamba_kernel(
    model_runner: GPUModelRunner, device: torch.device
) -> None:
    """Warm up postprocess_mamba_fused_kernel.

    This kernel is only used when speculative decoding is combined with a
    Mamba-hybrid model in "align" cache mode. The _dummy_run() warmup
    exercises Mamba forward layers but never triggers the postprocess step,
    so we must compile the kernel explicitly with the correct constexprs.
    """
    if not HAS_TRITON:
        return
    from vllm.v1.kv_cache_interface import MambaSpec
    from vllm.v1.worker.mamba_utils import postprocess_mamba_fused_kernel

    # Find the Mamba block_size from KV cache config.
    mamba_block_size = None
    for group in model_runner.kv_cache_config.kv_cache_groups:
        if isinstance(group.kv_cache_spec, MambaSpec):
            mamba_block_size = group.kv_cache_spec.block_size
            break
    if mamba_block_size is None:
        return

    # Minimal dummy tensors — one request, one state.
    # The kernel grid is (num_reqs, total_states). We use (1, 1) for
    # minimal compilation; only constexprs affect the compiled variant.
    num_reqs = 1
    total_states = 1

    num_accepted = torch.ones(num_reqs, dtype=torch.int32, device=device)
    mamba_state_idx = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    num_scheduled = torch.ones(num_reqs, dtype=torch.int32, device=device)
    num_computed = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    num_draft = torch.zeros(num_reqs, dtype=torch.int32, device=device)

    # Block table pointer array (one group).
    dummy_bt = torch.zeros(num_reqs, 2, dtype=torch.int32, device=device)
    block_table_ptrs = torch.tensor(
        [dummy_bt.data_ptr()], dtype=torch.int64, device=device
    )
    block_table_stride_req = dummy_bt.stride(0)

    # Per-state metadata (one state entry).
    # Allocate a small scratch buffer for the state copy target.
    scratch = torch.zeros(64, dtype=torch.uint8, device=device)
    state_base_addrs = torch.tensor(
        [scratch.data_ptr()], dtype=torch.int64, device=device
    )
    state_block_strides = torch.tensor([64], dtype=torch.int64, device=device)
    state_elem_sizes = torch.tensor([1], dtype=torch.int32, device=device)
    state_inner_sizes = torch.tensor([1], dtype=torch.int64, device=device)
    state_conv_widths = torch.tensor([0], dtype=torch.int32, device=device)
    state_group_indices = torch.tensor([0], dtype=torch.int32, device=device)

    num_accepted_out = torch.ones(num_reqs, dtype=torch.int32, device=device)

    grid = (num_reqs, total_states)
    postprocess_mamba_fused_kernel[grid](
        num_accepted,
        mamba_state_idx,
        num_scheduled,
        num_computed,
        num_draft,
        block_table_ptrs,
        block_table_stride_req,
        state_base_addrs,
        state_block_strides,
        state_elem_sizes,
        state_inner_sizes,
        state_conv_widths,
        state_group_indices,
        num_accepted_out,
        num_reqs,
        block_size=mamba_block_size,
        COPY_BLOCK_SIZE=1024,
    )
    logger.debug("Warmed up postprocess_mamba_fused_kernel.")
