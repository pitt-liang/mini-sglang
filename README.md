<p align="center">
<img width="400" src="/assets/logo.png">
</p>

# Mini-SGLang

A **lightweight yet high-performance** inference framework for Large Language Models.

---

Mini-SGLang is a compact implementation of [SGLang](https://github.com/sgl-project/sglang), designed to demystify the complexities of modern LLM serving systems. With a compact codebase of **~5,000 lines of Python**, it serves as both a capable inference engine and a transparent reference for researchers and developers.

## ✨ Key Features

- **High Performance**: Achieves state-of-the-art throughput and latency with advanced optimizations.
- **Lightweight & Readable**: A clean, modular, and fully type-annotated codebase that is easy to understand and modify.
- **Advanced Optimizations**:
  - **Radix Cache**: Reuses KV cache for shared prefixes across requests.
  - **Chunked Prefill**: Reduces peak memory usage for long-context serving.
  - **Overlap Scheduling**: Hides CPU scheduling overhead with GPU computation.
  - **Tensor Parallelism**: Scales inference across multiple GPUs.
  - **Optimized Kernels**: Integrates **FlashAttention** and **FlashInfer** for maximum efficiency.
  - ...

## 🚀 Quick Start

> **⚠️ Platform Support**: Mini-SGLang currently supports **Linux only** (x86_64 and aarch64). Windows and macOS are not supported due to dependencies on Linux-specific CUDA kernels (`sgl-kernel`, `flashinfer`). We recommend using [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) on Windows or Docker for cross-platform compatibility.

### 1. Environment Setup

We recommend using `uv` for a fast and reliable installation (note that `uv` does not conflict with `conda`).

```bash
# Run from the mini-sglang checkout; creates its own .venv from pyproject.toml/uv.lock.
uv sync --locked --python 3.12
source .venv/bin/activate
```

**Prerequisites**: This checkout pins the validated Torch 2.13 / CUDA 13.0 runtime
and Transformers 5.12.1. Use Python 3.12 (minimum 3.11), a CUDA-13-capable driver,
and a compatible **NVIDIA CUDA Toolkit** for JIT compilation. A virtual environment
isolates Python packages, not the host driver/toolkit. SGLang is not a runtime
dependency; do not activate or reuse `sglang/.venv-perf` to run Mini-SGLang.

### 2. Installation

Install Mini-SGLang directly from the source:

```bash
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang
uv sync --locked --python 3.12
source .venv/bin/activate
```

For tests and development, use `uv sync --locked --extra dev`. `uv sync` installs
this checkout in editable mode; `uv run --locked` or `.venv/bin/python` uses its
local environment without a `PYTHONPATH` override. Commit `uv.lock` together with
dependency changes. `sglang-kernel` (imported as `sgl_kernel`) is a standalone
operator package used by other backends, not the SGLang framework.
On aarch64, `uv pip check` reports an upstream wheel-tag mismatch for Torch's
pinned `nvidia-cusparselt-cu13==0.8.1`; its shared library is aarch64, but its
internal wheel tag says `sbsa`. The installed operator wheel also does not include
the legacy FA3 backend. This runtime has been validated for the GB200 Qwen4 path,
not all legacy GPU/backend combinations.

<details>
<summary><b>💡 Installing on Windows (WSL2)</b></summary>

Since Mini-SGLang requires Linux-specific dependencies, Windows users should use WSL2:

1. **Install WSL2** (if not already installed):
   ```powershell
   # In PowerShell (as Administrator)
   wsl --install
   ```

2. **Install CUDA on WSL2**:
   - Follow [NVIDIA's WSL2 CUDA guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
   - Ensure your Windows GPU drivers support WSL2

3. **Install Mini-SGLang in WSL2**:
   ```bash
   # Inside WSL2 terminal
   git clone https://github.com/sgl-project/mini-sglang.git
   cd mini-sglang
   uv sync --locked --python 3.12
   source .venv/bin/activate
   ```

4. **Access from Windows**: The server will be accessible at `http://localhost:8000` from Windows browsers and applications.

</details>

<details>
<summary><b>🐳 Running with Docker</b></summary>

**Prerequisites**:
- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

1. **Build the Docker image**:
   ```bash
   docker build -t minisgl .
   ```

2. **Run the server**:
   ```bash
   docker run --gpus all -p 1919:1919 \
       minisgl --model Qwen/Qwen3-0.6B --host 0.0.0.0
   ```

3. **Run in interactive shell mode**:
   ```bash
   docker run -it --gpus all \
       minisgl --model Qwen/Qwen3-0.6B --shell
   ```

4. **Using Docker Volumes for persistent caches** (recommended for faster subsequent startups):
   ```bash
   docker run --gpus all -p 1919:1919 \
       -v huggingface_cache:/app/.cache/huggingface \
       -v tvm_cache:/app/.cache/tvm-ffi \
       -v flashinfer_cache:/app/.cache/flashinfer \
       minisgl --model Qwen/Qwen3-0.6B --host 0.0.0.0
   ```

</details>

### 3. Online Serving

Launch an OpenAI-compatible API server with a single command.

```bash
# Deploy Qwen/Qwen3-0.6B on a single GPU
python -m minisgl --model "Qwen/Qwen3-0.6B"

# Deploy meta-llama/Llama-3.1-70B-Instruct on 4 GPUs with Tensor Parallelism, on port 30000
python -m minisgl --model "meta-llama/Llama-3.1-70B-Instruct" --tp 4 --port 30000
```

Once the server is running, you can send requests using standard tools like `curl` or any OpenAI-compatible client.

Qwen3.8-Flash-Next has an experimental native BF16 **text-only** path (GR, GDN,
QSA, shared/routed MoE and PLE). It supports batched CUDA Graph decode,
paged sparse prefill and FP32 recurrent state. The Qwen4 scheduler maps the default
`--cache-type radix` to a hybrid prefix cache; `--cache-type naive` disables reuse.
Vision, MTP and quantization are not supported. Optimized BF16 kernels are not bit-equivalent to
the reference path; set `MINISGL_QWEN4_REFERENCE=1` and pass
`--cuda-graph-max-bs 0` to select the eager reference.
A GB200 numerical-alignment path is enabled by default on SM100/CUDA 13+.
`MINISGL_QWEN4_SGLANG_NUMERICS=1` requires it explicitly; `=0` selects the older
fast path. This is separate from the older mini reference switch.
These switches are read into an immutable per-model policy at configuration time
and resolved on the worker's device before model construction and weight packing;
changing the environment afterwards does not switch an existing model or graph.
Alignment targets Torch dense GEMM, Triton MoE and FP32 recurrent state; stock
sparse top-k tie ordering is nondeterministic. This does not imply performance
parity or bit-exact outputs for every SGLang backend.

Example configuration for two GB200 GPUs (BF16 weights use approximately
166 GiB per rank before cache and workspace):

```bash
MINISGL_QWEN4_MNNVL=1 .venv/bin/python -m minisgl \
  --model-path /path/to/Qwen3.8-Flash-Next --tp-size 2 \
  --max-running-requests 4 --max-seq-len-override 8192 \
  --num-pages 512 --page-size 64 --cuda-graph-max-bs 4 --disable-pynccl
```

`MINISGL_QWEN4_MNNVL=1` enables optional NVLink multicast all-reduce. Run regression
tests with `python -m pytest tests/models tests/core/test_cache_allocate.py`.
Set `QWEN4_MODEL` to a local checkpoint and `QWEN4_HF_SOURCE` to an independent
Transformers `modeling_qwen4_exp.py` to enable their optional reference checks.
SGLang operator parity tests skip when the reference framework is unavailable;
it is not required in the production environment.

Qwen4 prefix reuse shares QSA KV/index pages and restores an immutable checkpoint
of **all** GDN recurrent/conv and PLE conv/token-history states into a private
request slot. Checkpoints are aligned to `lcm(page_size, 64)`, including a replay
boundary before the last prompt token. The aligned runtime records FP32 GDN
state inside a prefill chunk and gathers the matching GDN/PLE convolution windows,
without splitting that forward. Legacy/reference runtimes stop at the boundary.
Splitting a radix edge does not invent an intermediate recurrent state; matching
falls back to the deepest complete checkpoint. Decode state is not published.

`--qwen4-state-cache-mb 1024` caps the per-rank GPU checkpoint pool separately
from active request state. `--qwen4-checkpoint-interval 4096` controls periodic
prefill snapshots; the aligned prompt replay tail is also eligible. A KV hit
beyond the last state checkpoint identifies a shared-prefix junction: replay
captures its aligned boundary so later branches can resume there. The target
survives chunked scheduling. Internal tracking currently retains at most one
boundary per forward, prioritizing that junction, otherwise the deepest eligible
periodic/prompt boundary. For the
current BF16 checkpoint, one snapshot costs about 55.23 MiB/rank at TP2 (GDN
recurrent state stays FP32). Snapshot capacity is reserved before automatic KV
page sizing. Internal tracking also reserves one scratch state per running-request
capacity slot, which is included in automatic memory sizing. An exhausted protected
pool skips new snapshots instead of blocking
inference. PLE table offload/prefetch settings are independent of prefix caching.

Hybrid reuse enables `--qwen4-stable-numerics` by default: fixed prefill
projection/reduction geometry and deterministic QSA selection (exact FP32 score,
lowest index breaks ties, selected indices ordered by position). This avoids
batch-shape rounding changes and atomic top-k collection order changing outputs
when a prefix is removed. It is a different numerical policy from the original
SGLang-style atomic selector, not a claim of bitwise equivalence to SGLang.
The same flag can be enabled with `--cache-type naive` to isolate cache effects;
`--no-qwen4-stable-numerics` disables these selections for diagnostics. Ordinary
decode projection kernels remain unchanged. Non-final prefill chunks also end
on the 64-token recurrent grid; a batch's leftover budget must not shift later
requests' GDN chunk boundaries. Explicit total budgets below 64 still make
progress without this alignment. Kernel geometry alone does not guarantee
end-to-end invariance for all workloads.

Run the CPU lifecycle tests with
`.venv/bin/python -m pytest -o addopts='' tests/core/test_hybrid_cache.py`.
The real-weight regression `tests/models/smoke_qwen4_prefix.py` uses the actual
overlap scheduler to compare naive and hybrid logits/generation, repeated-prefix
and three-way branch hits, then measures TTFT without logit capture. Run it with
TP2 `torchrun`, `--model`, and an `--output` path outside the repository.
`--prompt-pattern text`, `--max-extend-tokens`, and `--branch-at` exercise textual
inputs, chunked scheduling, and a previously uncached shared-prefix junction.
The naive/hybrid comparison uses the same engine and numerical policy.
The JSON report includes TTFT, mean inter-token latency per request, hit lengths,
cache lifecycle counters and the runtime configuration. Use `--output-tokens 32`
or more for decode timing; the default four-token output is an accuracy smoke test.
`--controlled-topk` selects numerical-test ordering, not stock performance.
`--trace-components` records suffix activations from the first four layers (or
the layer IDs selected with `--trace-layers`) to localize numerical differences;
it is for accuracy diagnosis only. Prefix reuse
is still experimental: passing cache lifecycle/kernel tests does not establish
end-to-end logit equivalence or a production speedup. The real-weight regression
must pass its accuracy gates before treating either as validated.

PLE storage and lookup overlap are independently configurable before startup:

- `MINISGL_QWEN4_PLE_STORAGE=gpu` (default) keeps the BF16 table on the GPU.
  `=pinned` loads each TP shard directly into pinned CPU memory and uses GPU UVA
  lookup. It requires a CUDA device with mapped-host-memory/UVA support and enough
  host RAM: approximately 95.4 GiB total for this checkpoint, or 47.7 GiB per TP2
  rank. Only the embedding table is offloaded, not projections or request state.
- `MINISGL_QWEN4_PLE_PREFETCH=1` overlaps local lookup with the preceding decoder
  layer, for either storage backend. The default `=0` keeps lookup synchronous.
  TP reduction stays on the model stream; projection, gate and convolution retain
  their original ordering. Decode graph buffers are retained by batch size.

For host offload with overlap, prefix the command above with
`MINISGL_QWEN4_PLE_STORAGE=pinned MINISGL_QWEN4_PLE_PREFETCH=1`.
Host memory bandwidth/topology and stream contention affect performance; offload
does not guarantee lower latency on every device. No file-backed storage, CPU
lookup worker, FP8 PLE or cross-DP embedding sharding is implemented.

The implementation keeps model structure and eager/legacy/aligned dispatch in
`models/qwen4.py`, loading and packing in `models/qwen4_weight.py`, and GPU
operators in `kernel/qwen4_ops.py` (older numerics use the `legacy_` prefix).
`models/qwen4_ops.py` contains Torch reference helpers; GDN recurrence, sparse QSA,
attention metadata and cache storage retain their separate modules. Test-only
deterministic top-k and L2 helpers live under `tests/models`, not the runtime.

Without pytest, run the Qwen4 unit tests using the project environment:

```bash
.venv/bin/python -m unittest discover -s tests/models -p 'test_qwen4*.py' -v
```

For a real-weight TP2 smoke test on two GB200 GPUs:

```bash
smoke_dir=$(mktemp -d /tmp/minisgl-qwen4-smoke.XXXXXX)
MINISGL_QWEN4_REFERENCE=0 MINISGL_QWEN4_SGLANG_NUMERICS=1 \
MINISGL_QWEN4_MNNVL=1 OMP_NUM_THREADS=4 \
  .venv/bin/torchrun --standalone --nproc-per-node=2 tests/models/smoke_qwen4.py \
  --model /path/to/Qwen3.8-Flash-Next --output "$smoke_dir/current"
```

This checks finite logits, eager/graph equality, padded batches, repeated decode,
chunked scheduler prefill, cancellation and slot recycling. It uses controlled
top-k ordering to isolate regressions, not to certify stock-top-k determinism or
performance. Use `--compare /path/to/baseline-prefix` to require exact logits
against captures from a separate run; captures belong outside the repository.
Run it with all four PLE storage/prefetch combinations against the same baseline.
The unit suite also checks UVA rows, main-stream reduction, chunk/EOS history,
padding and repeated graph replay with reordered/recycled slots.

### 4. Interactive Shell

Chat with your model directly in the terminal by adding the `--shell` flag.

```bash
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell
```

![shell-example](https://lmsys.org/images/blog/minisgl/shell.png)

You can also use `/reset` to clear the chat history.

## Benchmark

### Offline inference

See [bench.py](./benchmark/offline/bench.py) for more details. Set `MINISGL_DISABLE_OVERLAP_SCHEDULING=1` for ablation study on overlap scheduling.

Test Configuration:

- Hardware: 1xH200 GPU.
- Model: Qwen3-0.6B, Qwen3-14B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100-1024 tokens
- Output Length: Randomly sampled between 100-1024 tokens

![offline](https://lmsys.org/images/blog/minisgl/offline.png)

### Online inference

See [benchmark_qwen.py](./benchmark/online/bench_qwen.py) for more details.

Test Configuration:

- Hardware: 4xH200 GPU, connected by NVLink.
- Model: Qwen3-32B
- Dataset: [Qwen trace](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/blob/main/qwen_traceA_blksz_16.jsonl), replaying first 1000 requests.

Launch command:

```bash
# Mini-SGLang
python -m minisgl --model "Qwen/Qwen3-32B" --tp 4 --cache naive

# SGLang
python3 -m sglang.launch_server --model "Qwen/Qwen3-32B" --tp 4 \
    --disable-radix --port 1919 --decode-attention flashinfer
```

> **Note**: If you encounter network issues when downloading models from HuggingFace, try using `--model-source modelscope` to download from ModelScope instead:
> ```bash
> python -m minisgl --model "Qwen/Qwen3-32B" --tp 4 --model-source modelscope
> ```

![online](https://lmsys.org/images/blog/minisgl/online.png)

## 📚 Learn More

- **[Detailed Features](./docs/features.md)**: Explore all available features and command-line arguments.
- **[System Architecture](./docs/structures.md)**: Dive deep into the design and data flow of Mini-SGLang.
