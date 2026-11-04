# lsa-sm70

Latent Sparse Attention (LSA) for NVIDIA V100 (SM70), packaged as an SGLang
attention backend.

LSA makes long-context attention subquadratic: keys are clustered in a learned
latent space, each query scores bucket centroids instead of every key, and the
read takes the selected keys plus a local window and sinks. This is a port of
the flat D-LSA design from
[Prefrontal-AI/latent-sparse-attention](https://github.com/Prefrontal-AI/latent-sparse-attention)
(arXiv 2610.42147, Apache-2.0) to the hardware SGLang cannot otherwise run it
on: V100 / SM70, through SGLang's Triton decode kernel.

## Status

Alpha. The path is validated end to end on Qwen3.8-27B-FP8 at TP4 on four
V100s: greedy output is token-identical to the dense `tilelang_fa_v100`
baseline on 32 tokens at 2,048 and 16,384 prompt tokens, with a mean absolute
log-probability shift of 6.7e-4 and 1.7e-4 nats.

What works:

- the flat D-LSA design: 512-key cluster blocks, Ward-clustered buckets, a
  512-key window and 4 sinks, learned `[768, 64]` query/key maps;
- grouped-query models (24 query / 4 KV heads), tensor parallel with the index
  maps sharded across the attention TP group;
- prefill and decode, bf16 or fp16 KV caches, `--page-size 1`.

What does not:

- CUDA graph capture: the selection is host-orchestrated, so serve with
  `--disable-cuda-graph`;
- speculative decoding, data-parallel attention, context parallel;
- an fp8 KV cache (the index projects from the pool);
- speed tuning: selection is two Triton programs a row, not the reference's
  fused scan.

## Install

The package expects an SGLang environment matching the pin it was written
against (`sgl-project/sglang` v0.5.22, `baa9befa`), Python 3.12, torch and
Triton.

```sh
pip install lsa-sm70
```

Installing registers the `sglang.srt.plugins` entry point. Select the backend
by name:

```sh
SGLANG_RESTOMOD_LSA_MAP=/path/to/qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt \
python -m sglang.launch_server \
  --model-path /path/to/Qwen3.8-27B-FP8 \
  --tp-size 4 --dtype float16 --quantization fp8 \
  --attention-backend lsa_sm70 \
  --page-size 1 --disable-cuda-graph \
  --linear-attn-prefill-backend triton --linear-attn-decode-backend triton
```

The D-LSA map is the released projection file for Qwen3.8-27B, 16 layers
(3, 7, ..., 63) of `[768, 64]` `W_Q_idx`/`W_K_idx`. It is not distributed here;
download it from the reference release and point `SGLANG_RESTOMOD_LSA_MAP` at
it.

Design knobs, with the paper's defaults:

| variable | default |
|---|---:|
| `SGLANG_RESTOMOD_LSA_TOP_K` | 2048 |
| `SGLANG_RESTOMOD_LSA_TOP_P` | 512 |
| `SGLANG_RESTOMOD_LSA_WINDOW` | 512 |
| `SGLANG_RESTOMOD_LSA_SINK` | 4 |

## How it works

| module | what it does |
|---|---|
| `index_space.py` | position-free dims, grouped-query collapse, learned projection |
| `ward.py` | bottom-up Ward clustering of a cluster block's pools |
| `reference.py` | the flat design in plain torch - the oracle |
| `design.py` | the design's knobs and a request's bucket table |
| `selection.py` | the oracle's selection vectorized, or the Triton programs on CUDA |
| `kernels.py` | `_scan_top_p` and `_expand`, the two SM70 selection programs |
| `read.py` | the sparse read through `decode_attention_fwd` |
| `backend.py` | `LsaSm70Backend(AttentionBackend)` |
| `backend_registry.py` | registers `--attention-backend lsa_sm70` |

The index is kept per `(layer, request)`: closed 512-key blocks contribute
centroids and members, the open block keeps its projected keys, and a decode
appends one key. Every rank projects its own KV heads through its rows of the
maps and the partials are all-reduced, so every rank selects the same lists.

## Tests

```sh
pytest -q -m "not gpu"     # CPU: the oracle, the selection, the bucket table
pytest -q -m gpu           # SM70: kernels and read against the oracle
```

The GPU tests skip unless the device is SM70.

## Provenance

The selection core is vendored from
`Prefrontal-AI/latent-sparse-attention` @ `97b82d6` (Apache-2.0); the serving
path and the Triton kernels are written here. `PROVENANCE.md` lists every file
and what changed. The reference's CuTeDSL kernels are Hopper/Blackwell only and
are not included.

## License

Apache-2.0. See `LICENSE` and `PROVENANCE.md`.
