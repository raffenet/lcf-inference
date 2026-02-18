# Frontier

Frontier is an AMD Instinct MI250X system at the Oak Ridge Leadership Computing Facility, managed with Slurm.

!!! warning
    Frontier support in Aegis is planned but not yet implemented. The instructions below cover manual vLLM setup on Frontier nodes.

## Environment Setup

Until native vLLM support is confirmed on Frontier, use the ROCm vLLM Docker images via Apptainer:

```bash
apptainer build vllm-openai-rocm.sif docker://vllm/vllm-openai-rocm
apptainer shell vllm-openai-rocm.sif
vllm --version
```

## Model Weights

Set HTTP proxy settings for outbound access when downloading models from Hugging Face:

```bash
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'
```

!!! note
    Some models hosted on Hugging Face are gated and require a [Hugging Face authentication token](https://huggingface.co/docs/hub/en/security-tokens):

    ```bash
    export HF_TOKEN=<your_token>
    ```

We recommend using Frontier's Burst Buffer storage for staging model weights.

## Serving Models

### Small Models (Single GPU)

Models with fewer than 7 billion parameters typically fit within a single GPU's 64 GB memory:

```bash
vllm serve meta-llama/Llama-2-7b-chat-hf
```

### Medium Models (Multiple GPUs, Single Node)

For tensor parallelism across multiple GPUs (`TP>1`), first configure a Ray cluster:

```bash
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
unset ROCM_VISIBLE_DEVICES  # vLLM throws an error if it sees this envvar
ray start --head --node-ip-address=$VLLM_HOST_IP --num-cpus=128 --num-gpus=8 &
```

Then serve the model:

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --max-model-len 32768
```

### Large Models (Multiple Nodes)

Multi-node serving on Frontier is not yet documented. This will be updated as Aegis adds Slurm/Frontier support.
