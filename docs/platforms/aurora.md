# Aurora

Aurora is an Intel Data Center GPU Max (Ponte Vecchio) system at the Argonne Leadership Computing Facility, managed with PBS.

## Environment Setup

Load the frameworks module which includes an XPU-enabled vLLM installation:

```bash
module load frameworks
export NUMEXPR_MAX_THREADS=208  # number of CPU threads on Aurora node
export CCL_PROCESS_LAUNCHER=torchrun
vllm --version
```

## Model Weights

Model weights for commonly used open-weight models are available in `/flare/datasets/model-weights/hub` on Aurora. However, accessing these weights from more than a single GPU can be very slow. For multi-GPU scenarios, read the weights once from the parallel filesystem and distribute them to node-local storage with MPI broadcast:

```bash
export HF_HOME="/tmp/hf_home"
mpiexec -ppn 1 ./bcast /flare/datasets/model-weights/hub/models--meta-llama--Llama-3.3-70B-Instruct /tmp/hf_home/hub
```

!!! note
    Some models hosted on Hugging Face are gated and require a [Hugging Face authentication token](https://huggingface.co/docs/hub/en/security-tokens):

    ```bash
    export HF_TOKEN="YOUR_HF_TOKEN"
    ```

## Serving Models

### Small Models (Single GPU Tile)

Models with fewer than 7 billion parameters typically fit within a single tile's 64 GB memory. No additional configuration is required:

```bash
module load frameworks
export NUMEXPR_MAX_THREADS=208
export CCL_PROCESS_LAUNCHER=torchrun
export HF_HOME=/flare/datasets/model-weights
export HF_TOKEN=<your_token>
vllm serve meta-llama/Llama-2-7b-chat-hf
```

### Medium Models (Multiple GPU Tiles)

For models up to ~70B parameters, use tensor parallelism across tiles on a single node:

```bash
module load frameworks
export NUMEXPR_MAX_THREADS=208
export CCL_PROCESS_LAUNCHER=torchrun
export HF_HOME=/tmp/hf_home
export HF_TOKEN=<your_token>
# Stage weights to local storage
mpiexec -ppn 1 ./bcast /flare/datasets/model-weights/hub/models--meta-llama--Llama-3.3-70B-Instruct /tmp/hf_home/hub
vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 8
```

### Large Models (Multiple Nodes)

For models that exceed single-node capacity, Aegis handles multi-node orchestration automatically. Configure `tensor_parallel_size` to exceed the GPUs on a single node (12 on Aurora) and Aegis will allocate the necessary nodes:

```yaml
model: large-model/name
tensor_parallel_size: 24  # spans 2 nodes
instances: 1
```

## Using Aegis on Aurora

With Aegis, the environment setup, weight staging, and instance orchestration are handled automatically. Create a config file and submit:

```bash
aegis submit --config config.yaml
```

See [Getting Started](../getting-started.md) for a full walkthrough.
