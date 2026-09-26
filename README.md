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
QSA, shared/routed MoE and GPU-resident PLE). It supports batched CUDA Graph decode,
paged sparse prefill and FP32 recurrent state. Prefix reuse, vision, MTP and
quantization are not supported. Optimized BF16 kernels are not bit-equivalent to
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
