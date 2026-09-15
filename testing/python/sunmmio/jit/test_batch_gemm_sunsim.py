import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import tilelang
import tilelang.language as T
from tilelang.layout import make_mxznz_layout, make_mxzz_layout, make_zz_layout


MESH = (4, 4)


@dataclass(frozen=True)
class BatchGemmCase:
    name: str
    batch: int = 2
    parent_batch: int | None = None
    batch_begin: int = 0
    M: int = 16
    N: int = 32
    K: int = 32
    a_batched: bool = True
    b_batched: bool = True
    output_batched: bool = True
    transpose_b: bool = False
    clear_accum: bool = True
    accum_dtype: str = "float32"
    version: int = 2
    a_dtype: str = "bfloat16"
    b_dtype: str = "bfloat16"


SMOKE_CASES = (
    BatchGemmCase("v2-rank3-m16"),
    BatchGemmCase("v1-reduction-m16", output_batched=False, version=1),
)


BROAD_CASES = (
    *SMOKE_CASES,
    BatchGemmCase("v1-rank3-m16", version=1),
    BatchGemmCase("v2-reduction-m16", output_batched=False),
    BatchGemmCase("v2-rank3-shared-a", batch=4, a_batched=False),
    BatchGemmCase("v2-rank3-shared-w", batch=4, b_batched=False),
    BatchGemmCase(
        "v2-rank3-shared-a-w", batch=4, a_batched=False, b_batched=False
    ),
    BatchGemmCase(
        "v2-reduction-shared-a", batch=4, a_batched=False, output_batched=False
    ),
    BatchGemmCase(
        "v2-reduction-shared-w", batch=4, b_batched=False, output_batched=False
    ),
    BatchGemmCase("v1-rank3-trans-b", M=32, transpose_b=True, version=1),
    BatchGemmCase(
        "v2-reduction-trans-b", M=32, transpose_b=True, output_batched=False
    ),
    BatchGemmCase("v1-rank3-accumulate", M=32, clear_accum=False, version=1),
    BatchGemmCase(
        "v2-reduction-accumulate", M=32, output_batched=False, clear_accum=False
    ),
    BatchGemmCase("v2-rank3-batch1", batch=1, M=32),
    BatchGemmCase("v1-rank3-batch4", batch=4, M=32, version=1),
    BatchGemmCase("v2-rank3-batch8", batch=8),
    BatchGemmCase("v1-rank3-m64", M=64, version=1),
    BatchGemmCase("v2-rank3-n64", M=32, N=64),
    BatchGemmCase("v2-rank3-k64", M=32, K=64),
    BatchGemmCase("v2-rank3-64-cube", M=64, N=64, K=64),
    BatchGemmCase("v2-rank3-bf16-output", M=32, accum_dtype="bfloat16"),
    BatchGemmCase(
        "v1-reduction-bf16-output",
        M=32,
        output_batched=False,
        accum_dtype="bfloat16",
        version=1,
    ),
)


MX_CASES = (
    BatchGemmCase(
        "v2-rank3-bf16-mxfp8-trans-b",
        M=32,
        K=64,
        transpose_b=True,
        b_dtype="mxfp8",
    ),
    BatchGemmCase(
        "v1-reduction-bf16-mxfp8-trans-b",
        M=32,
        K=64,
        output_batched=False,
        transpose_b=True,
        b_dtype="mxfp8",
        version=1,
    ),
    BatchGemmCase(
        "v2-rank3-mxfp8-mxfp8-trans-b",
        M=32,
        K=64,
        transpose_b=True,
        a_dtype="mxfp8",
        b_dtype="mxfp8",
    ),
    BatchGemmCase(
        "v2-rank3-bf16-mxfp4-mn",
        M=32,
        K=64,
        b_dtype="mxfp4",
    ),
)

PARTIAL_CASES = (
    BatchGemmCase(
        "v2-rank3-partial-b1-e2-of4",
        parent_batch=4,
        batch_begin=1,
        M=32,
    ),
    BatchGemmCase(
        "v1-reduction-partial-b1-e2-of4",
        parent_batch=4,
        batch_begin=1,
        M=32,
        output_batched=False,
        version=1,
    ),
)

ALL_CASES = (*BROAD_CASES, *MX_CASES, *PARTIAL_CASES)


def _enable_legacy_toolchain_compat():
    from tilelang.jit.adapter.sunmmio import libgen

    render = libgen._render_sunsim_main_thunk

    def render_with_smaller_host_descriptor_area(kernel_name):
        source = render(kernel_name)
        old = "_descriptor_start[4096] = {1};"
        if old not in source:
            raise RuntimeError("unexpected Sunsim main thunk template")
        return source.replace(old, "_descriptor_start[3072] = {1};")

    libgen._render_sunsim_main_thunk = render_with_smaller_host_descriptor_area


def _batch_gemm_prim(
    batch,
    parent_batch,
    batch_begin,
    M,
    N,
    K,
    a_batched,
    b_batched,
    output_batched,
    transpose_b,
    clear_accum,
    accum_dtype,
    version,
    a_dtype,
    b_dtype,
):
    dtype_by_name = {"mxfp8": T.mxfp8, "mxfp4": T.mxfp4}
    a_tl_dtype = dtype_by_name.get(a_dtype, a_dtype)
    b_tl_dtype = dtype_by_name.get(b_dtype, b_dtype)
    allocation_batch = parent_batch if parent_batch is not None else batch
    partial_batch = batch_begin != 0 or allocation_batch != batch
    a_shape = (allocation_batch, M, K) if a_batched else (M, K)
    b_matrix_shape = (N, K) if transpose_b else (K, N)
    b_shape = (
        (allocation_batch, *b_matrix_shape) if b_batched else b_matrix_shape
    )
    c_shape = (allocation_batch, M, N) if output_batched else (M, N)
    placement = T.placement.replicated()
    a_axes = (len(a_shape) - 2, len(a_shape) - 1)
    b_axes = (len(b_shape) - 2, len(b_shape) - 1)
    a_layout = (
        make_mxzz_layout(a_shape, a_axes, dtype=a_tl_dtype)
        if a_dtype in ("mxfp8", "mxfp4")
        else make_zz_layout(a_shape, a_axes, (32, 32))
    )
    b_layout = (
        make_mxzz_layout(b_shape, b_axes, dtype=b_tl_dtype)
        if b_dtype in ("mxfp8", "mxfp4") and transpose_b
        else make_mxznz_layout(b_shape, b_axes, dtype=b_tl_dtype)
        if b_dtype in ("mxfp8", "mxfp4")
        else make_zz_layout(b_shape, b_axes, (32, 32))
    )
    c_layout = make_zz_layout(
        c_shape, (len(c_shape) - 2, len(c_shape) - 1), (32, 32)
    )

    @T.prim_func
    def main(
        A: T.MeshTensor(a_shape, placement, a_tl_dtype, layout=a_layout),
        B: T.MeshTensor(b_shape, placement, b_tl_dtype, layout=b_layout),
        C: T.MeshTensor(c_shape, placement, accum_dtype, layout=c_layout),
    ):
        with T.Kernel():
            a_shared = T.alloc_shared(a_shape, a_tl_dtype)
            b_shared = T.alloc_shared(b_shape, b_tl_dtype)
            c_shared = T.alloc_shared(c_shape, accum_dtype)

            if a_batched:
                T.copy(A[0, 0, 0], a_shared)
            else:
                T.copy(A[0, 0], a_shared)
            if b_batched:
                T.copy(B[0, 0, 0], b_shared)
            else:
                T.copy(B[0, 0], b_shared)
            if not clear_accum or (output_batched and partial_batch):
                if output_batched:
                    T.copy(C[0, 0, 0], c_shared)
                else:
                    T.copy(C[0, 0], c_shared)
            batch_end = batch_begin + batch
            gemm = T.gemm_v1 if version == 1 else T.gemm_v2
            if partial_batch:
                if a_batched:
                    if b_batched:
                        if output_batched:
                            gemm(
                                a_shared[batch_begin:batch_end, :, :],
                                b_shared[batch_begin:batch_end, :, :],
                                c_shared[batch_begin:batch_end, :, :],
                                transpose_B=transpose_b,
                                clear_accum=clear_accum,
                            )
                        else:
                            gemm(
                                a_shared[batch_begin:batch_end, :, :],
                                b_shared[batch_begin:batch_end, :, :],
                                c_shared,
                                transpose_B=transpose_b,
                                clear_accum=clear_accum,
                            )
                    elif output_batched:
                        gemm(
                            a_shared[batch_begin:batch_end, :, :],
                            b_shared,
                            c_shared[batch_begin:batch_end, :, :],
                            transpose_B=transpose_b,
                            clear_accum=clear_accum,
                        )
                    else:
                        gemm(
                            a_shared[batch_begin:batch_end, :, :],
                            b_shared,
                            c_shared,
                            transpose_B=transpose_b,
                            clear_accum=clear_accum,
                        )
                elif b_batched:
                    if output_batched:
                        gemm(
                            a_shared,
                            b_shared[batch_begin:batch_end, :, :],
                            c_shared[batch_begin:batch_end, :, :],
                            transpose_B=transpose_b,
                            clear_accum=clear_accum,
                        )
                    else:
                        gemm(
                            a_shared,
                            b_shared[batch_begin:batch_end, :, :],
                            c_shared,
                            transpose_B=transpose_b,
                            clear_accum=clear_accum,
                        )
                else:
                    gemm(
                        a_shared,
                        b_shared,
                        c_shared[batch_begin:batch_end, :, :],
                        transpose_B=transpose_b,
                        clear_accum=clear_accum,
                    )
            else:
                gemm(
                    a_shared,
                    b_shared,
                    c_shared,
                    transpose_B=transpose_b,
                    clear_accum=clear_accum,
                )
            if output_batched:
                T.copy(c_shared, C[0, 0, 0])
            else:
                T.copy(c_shared, C[0, 0])

    return main


@tilelang.jit(target="sunmmio", execution_backend="sunmmio_sunsim")
def batch_gemm_sunsim(
    batch,
    parent_batch,
    batch_begin,
    M,
    N,
    K,
    a_batched,
    b_batched,
    output_batched,
    transpose_b,
    clear_accum,
    accum_dtype,
    version,
    a_dtype,
    b_dtype,
):
    return _batch_gemm_prim(
        batch,
        parent_batch,
        batch_begin,
        M,
        N,
        K,
        a_batched,
        b_batched,
        output_batched,
        transpose_b,
        clear_accum,
        accum_dtype,
        version,
        a_dtype,
        b_dtype,
    )


def _run_case(case, timeout=600.0):
    import ml_dtypes
    import numpy as np
    import sunsim

    rng = np.random.default_rng(1000 + sum(case.name.encode("ascii")))
    mx_dtypes = {"mxfp8", "mxfp4"}
    uses_mx = case.a_dtype in mx_dtypes or case.b_dtype in mx_dtypes
    if uses_mx:
        npuir_cases = Path(__file__).resolve().parents[4] / "3rdparty/NPU-IR/test/gem5/cases"
        if str(npuir_cases) not in sys.path:
            sys.path.insert(0, str(npuir_cases))
        import mx_ref

    allocation_batch = (
        case.parent_batch if case.parent_batch is not None else case.batch
    )
    partial_batch = case.batch_begin != 0 or allocation_batch != case.batch
    batch_slice = slice(case.batch_begin, case.batch_begin + case.batch)
    a_shape = (
        (allocation_batch, case.M, case.K)
        if case.a_batched
        else (case.M, case.K)
    )
    b_matrix_shape = (
        (case.N, case.K) if case.transpose_b else (case.K, case.N)
    )
    b_shape = (
        (allocation_batch, *b_matrix_shape)
        if case.b_batched
        else b_matrix_shape
    )
    if case.a_dtype in mx_dtypes:
        a_blobs = []
        a_values = []
        for _ in range(allocation_batch if case.a_batched else 1):
            codes, scales = mx_ref.random_codes_scales(
                rng, case.M, case.K, case.a_dtype
            )
            a_blobs.append(mx_ref.pack_mxzz(codes, scales, case.a_dtype))
            a_values.append(mx_ref.dequant(codes, scales))
        a_input = np.concatenate(a_blobs)
        a_reference = np.stack(a_values) if case.a_batched else a_values[0]
    else:
        a_input = rng.uniform(-0.25, 0.25, a_shape).astype(ml_dtypes.bfloat16)
        a_reference = a_input.astype(np.float32)

    if case.b_dtype in mx_dtypes:
        b_blobs = []
        b_values = []
        for _ in range(allocation_batch if case.b_batched else 1):
            if case.transpose_b:
                codes, scales = mx_ref.random_codes_scales(
                    rng, case.N, case.K, case.b_dtype
                )
                b_blobs.append(mx_ref.pack_mxzz(codes, scales, case.b_dtype))
                b_values.append(mx_ref.dequant(codes, scales))
            else:
                codes, scales = mx_ref.random_weight_kn(
                    rng, case.K, case.N, case.b_dtype
                )
                b_blobs.append(
                    mx_ref.pack_mxznz_weight(codes, scales, case.b_dtype)
                )
                b_values.append(mx_ref.dequant_kn(codes, scales))
        b_input = np.concatenate(b_blobs)
        b_reference = np.stack(b_values) if case.b_batched else b_values[0]
    else:
        b_input = rng.uniform(-0.25, 0.25, b_shape).astype(ml_dtypes.bfloat16)
        b_reference = b_input.astype(np.float32)

    a_for_reference = (
        a_reference[batch_slice] if case.a_batched else a_reference
    )
    b_for_reference = (
        b_reference[batch_slice] if case.b_batched else b_reference
    )
    if case.transpose_b:
        b_for_reference = np.swapaxes(b_for_reference, -1, -2)
    products = np.matmul(a_for_reference, b_for_reference)

    output_dtype = (
        np.float32 if case.accum_dtype == "float32" else ml_dtypes.bfloat16
    )
    output_shape = (
        (allocation_batch, case.M, case.N)
        if case.output_batched
        else (case.M, case.N)
    )
    initial = rng.uniform(-0.25, 0.25, output_shape).astype(output_dtype)
    if case.output_batched:
        expected = initial.astype(np.float32).copy()
        active = np.broadcast_to(products, (case.batch, case.M, case.N)).copy()
        if not case.clear_accum:
            active += initial[batch_slice].astype(np.float32)
        expected[batch_slice] = active
    else:
        expected = products.sum(axis=0) if products.ndim == 3 else products.copy()
        if not case.clear_accum:
            expected += initial.astype(np.float32)
    if case.accum_dtype == "bfloat16":
        expected = expected.astype(ml_dtypes.bfloat16).astype(np.float32)

    placement = [sunsim.R(), sunsim.R()]
    a_arg = (
        sunsim.Input(a_input)
        if case.a_dtype in mx_dtypes
        else sunsim.Input(
            a_input,
            placement=placement,
            layout=sunsim.Layout.zz(
                block_dims=(1, 2) if case.a_batched else (0, 1)
            ),
        )
    )
    b_arg = (
        sunsim.Input(b_input)
        if case.b_dtype in mx_dtypes
        else sunsim.Input(
            b_input,
            placement=placement,
            layout=sunsim.Layout.zz(
                block_dims=(1, 2) if case.b_batched else (0, 1)
            ),
        )
    )
    c_layout = sunsim.Layout.zz(
        block_dims=(1, 2) if case.output_batched else (0, 1)
    )
    if case.clear_accum and not (case.output_batched and partial_batch):
        output = sunsim.Output(
            expected.shape, output_dtype, placement=placement, layout=c_layout
        )
    else:
        output = sunsim.Inout(initial, placement=placement, layout=c_layout)
    kernel = batch_gemm_sunsim(
        case.batch,
        case.parent_batch,
        case.batch_begin,
        case.M,
        case.N,
        case.K,
        case.a_batched,
        case.b_batched,
        case.output_batched,
        case.transpose_b,
        case.clear_accum,
        case.accum_dtype,
        case.version,
        case.a_dtype,
        case.b_dtype,
    )

    start = time.perf_counter()
    result = kernel(
        a_arg,
        b_arg,
        output,
        mesh=MESH,
        timeout=timeout,
    )
    wall_seconds = time.perf_counter() - start
    actual = output.data.astype(np.float32)
    atol = 5e-2 if uses_mx else 3e-2 if case.accum_dtype == "bfloat16" else 1e-2
    rtol = 2e-2 if uses_mx else 1e-2
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    max_abs_error = float(np.max(np.abs(actual - expected)))
    cycles = result.stats.cycles if result.stats is not None else None
    record = {
        **asdict(case),
        "status": "PASS",
        "max_abs_error": max_abs_error,
        "cycles": cycles,
        "wall_seconds": wall_seconds,
    }
    print(
        f"Batch GEMM PASS: case={case.name}, version={case.version}, "
        f"shape=({case.batch},{case.M},{case.N},{case.K}), "
        f"parent_batch={allocation_batch}, batch_begin={case.batch_begin}, "
        f"max_abs_error={max_abs_error:.6f}, cycles={cycles}, "
        f"wall_seconds={wall_seconds:.3f}"
    )
    return record


def _git_revision(path):
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def _write_results(
    path, started_at, records, tilelang_revision=None, npu_ir_revision=None
):
    repo_root = Path(__file__).resolve().parents[4]
    payload = {
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "mesh": list(MESH),
        "tilelang_commit": tilelang_revision or _git_revision(repo_root),
        "npu_ir_commit": npu_ir_revision
        or _git_revision(repo_root / "3rdparty" / "NPU-IR"),
        "cases": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_batch_gemm_v2_rank3_m16_sunsim():
    _run_case(SMOKE_CASES[0])


def test_batch_gemm_v1_reduction_m16_sunsim():
    _run_case(SMOKE_CASES[1])


def test_runtime_batch_value_is_part_of_jit_specialization_key():
    common = (
        None,
        0,
        32,
        32,
        32,
        True,
        True,
        True,
        False,
        True,
        "float32",
        2,
        "bfloat16",
        "bfloat16",
    )
    key_b1 = batch_gemm_sunsim.parse_cache_key(1, *common)
    key_b2 = batch_gemm_sunsim.parse_cache_key(2, *common)
    assert key_b1 != key_b2
    assert key_b2 == batch_gemm_sunsim.parse_cache_key(2, *common)

    tir_b1 = batch_gemm_sunsim.get_tir(1, *common)
    tir_b2 = batch_gemm_sunsim.get_tir(2, *common)
    assert int(tir_b1.buffer_map[tir_b1.params[0]].shape[0]) == 1
    assert int(tir_b2.buffer_map[tir_b2.params[0]].shape[0]) == 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=("smoke", "broad", "mx", "partial", "all"),
        default="smoke",
    )
    parser.add_argument("--case", action="append", dest="case_names")
    parser.add_argument("--results-json", type=Path)
    parser.add_argument("--tilelang-revision")
    parser.add_argument("--npu-ir-revision")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--legacy-toolchain-compat",
        action="store_true",
        help="reserve 1 KiB for the pre-v0.2.7 ODMA scratch section",
    )
    args = parser.parse_args()
    if args.legacy_toolchain_compat:
        _enable_legacy_toolchain_compat()

    cases = {
        "smoke": SMOKE_CASES,
        "broad": BROAD_CASES,
        "mx": MX_CASES,
        "partial": PARTIAL_CASES,
        "all": ALL_CASES,
    }[args.suite]
    if args.case_names:
        requested = set(args.case_names)
        cases = tuple(case for case in ALL_CASES if case.name in requested)
        missing = requested - {case.name for case in cases}
        if missing:
            parser.error(f"unknown case(s): {', '.join(sorted(missing))}")

    started_at = datetime.now(timezone.utc).isoformat()
    records = []
    for case in cases:
        try:
            records.append(_run_case(case, timeout=args.timeout))
        except Exception as error:
            records.append({**asdict(case), "status": "FAIL", "error": str(error)})
            print(f"Batch GEMM FAIL: case={case.name}: {error}", flush=True)
        if args.results_json:
            _write_results(
                args.results_json,
                started_at,
                records,
                tilelang_revision=args.tilelang_revision,
                npu_ir_revision=args.npu_ir_revision,
            )

    passed = sum(record["status"] == "PASS" for record in records)
    print(f"Batch GEMM summary: {passed}/{len(records)} passed")
    if passed != len(records):
        raise SystemExit(1)
