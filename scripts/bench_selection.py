#!/usr/bin/env python3
"""Measure one decode step's LSA selection path on an SM70 device.

Stages, timed separately with CUDA events:

- ``index``  - append one projected key and rebuild the bucket table
               (``RequestIndex.append`` + ``RequestIndex.buckets``), which the
               backend does once a layer a step;
- ``select`` - the two Triton programs over the batch's query rows;
- ``read``   - ``lsa_sparse_read`` over the selected lists.

``--graph`` also captures ``select`` + ``read`` in one ``torch.cuda.CUDAGraph``
and checks that a replay follows a changed input and matches the eager result.
The index stage is not captured: it is the host-side part a capturable backend
has to move to the device.

    PYTHONPATH=/homes/pding/pin-site \\
        python scripts/bench_selection.py --context 8192 32768 --batch 1 4

The geometry is the Qwen3.8-27B TP4 one: one KV head a rank and six query
heads, a 256-dim head, and the paper's 2048/512/4 budget.
"""

from __future__ import annotations

import argparse
import statistics
import sys

import torch

from lsa_sm70.design import LsaDesign, RequestIndex
from lsa_sm70.read import lsa_sparse_read
from lsa_sm70.selection import select_lists

INDEX_DIM = 64
KV_HEADS = 1
QUERY_HEADS = 6
HEAD_DIM = 256
SCALE = HEAD_DIM**-0.5


def _unit(rows: int, dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    values = torch.randn((rows, dim), generator=generator, device="cuda")
    return values / values.norm(dim=1, keepdim=True)


def _timed(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
    return statistics.median(samples)


def _build_index(design: LsaDesign, context: int) -> RequestIndex:
    index = RequestIndex(design, torch.device("cuda"))
    index.append(_unit(context, INDEX_DIM, seed=0))
    return index


def _step(select_queries, positions, buckets, attention_queries, keys, values, design: LsaDesign):
    lists, lengths = select_lists(
        select_queries,
        positions,
        buckets,
        top_p=design.top_p,
        top_k=design.top_k,
        sink_size=design.sink_size,
        window_size=design.window_size,
    )
    return lsa_sparse_read(attention_queries, keys, values, lists, lengths, SCALE)


def _graph_check(step, mutate, *, warmup: int, iters: int) -> None:
    """Capture ``step``; then prove a replay follows a changed input."""
    try:
        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = step()
        graph.replay()
        torch.cuda.synchronize()
        before = out.clone()
        mutate()
        graph.replay()
        torch.cuda.synchronize()
        after = out.clone()
        followed = not torch.equal(before, after)
        eager = step()
        matches = torch.allclose(after, eager, atol=1e-3, rtol=1e-3)
    except Exception as exc:  # noqa: BLE001 - the point is to report the failure
        print(f"  graph: not capturable - {type(exc).__name__}: {exc}")
        return

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        graph.replay()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
    print(
        f"  graph: captured, replay {statistics.median(samples):.3f} ms, "
        f"follows input {followed}, eager match {matches}"
    )


def _case(design: LsaDesign, context: int, batch: int, args: argparse.Namespace) -> None:
    """One (context, batch) row: time the three stages and optionally capture."""
    index = _build_index(design, context)
    buckets = index.buckets()
    keys = _unit(context + 1, HEAD_DIM, seed=1).reshape(context + 1, KV_HEADS, HEAD_DIM)
    values = _unit(context + 1, HEAD_DIM, seed=2).reshape(context + 1, KV_HEADS, HEAD_DIM)
    next_key = _unit(1, INDEX_DIM, seed=3)
    queries = _unit(batch, INDEX_DIM, seed=4)
    positions = torch.full((batch,), context - 1, dtype=torch.int64, device="cuda")
    attention_queries = _unit(batch * QUERY_HEADS, HEAD_DIM, seed=5).reshape(
        batch, QUERY_HEADS, HEAD_DIM
    )

    def index_step():
        # One appended key, then the table rebuild the backend does a layer.
        index.append(next_key)
        index.buckets()
        index.open.pop()
        index.open_count -= 1

    def select_step():
        return select_lists(
            queries,
            positions,
            buckets,
            top_p=design.top_p,
            top_k=design.top_k,
            sink_size=design.sink_size,
            window_size=design.window_size,
        )

    lists, lengths = select_step()

    def read_step():
        return lsa_sparse_read(attention_queries, keys, values, lists, lengths, SCALE)

    def step():
        return _step(queries, positions, buckets, attention_queries, keys, values, design)

    def mutate():
        positions.add_(1)

    index_ms = _timed(index_step, warmup=args.warmup, iters=args.iters)
    select_ms = _timed(select_step, warmup=args.warmup, iters=args.iters)
    read_ms = _timed(read_step, warmup=args.warmup, iters=args.iters)
    total = index_ms + select_ms + read_ms
    print(
        f"{context:>7} {batch:>3} {index_ms:>9.3f} {select_ms:>10.3f} "
        f"{read_ms:>8.3f} {total:>9.3f} {100 * index_ms / total:>7.1f} "
        f"{100 * select_ms / total:>8.1f} {100 * read_ms / total:>6.1f}"
    )
    if args.graph:
        _graph_check(step, mutate, warmup=args.warmup, iters=20)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, nargs="+", default=[8192, 32768])
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--graph", action="store_true", help="also try CUDA graph capture")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 1
    design = LsaDesign()
    print(
        f"design: top_k={design.top_k} top_p={design.top_p} window={design.window_size} "
        f"sink={design.sink_size} width={design.width}"
    )
    print(
        f"{'ctx':>7} {'bs':>3} {'index_ms':>9} {'select_ms':>10} {'read_ms':>8} "
        f"{'total_ms':>9} {'index%':>7} {'select%':>8} {'read%':>6}"
    )

    for context in args.context:
        if context % design.cluster_block_size:
            print(f"skip {context}: not a multiple of {design.cluster_block_size}")
            continue
        for batch in args.batch:
            _case(design, context, batch, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
