# Inference with vLLM on Aurora

## Load frameworks module with XPU-enabled vLLM installation

```bash linenums="1"
module load frameworks
export NUMEXPR_MAX_THREADS=208 # number of CPU threads on Aurora node
export CCL_PROCESS_LAUNCHER=torchrun
vllm --version
```

## Example output from Aurora compute node

```console
raffenet@x4311c1s4b0n0:~> module load frameworks
(/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.2.0) raffenet@x4311c1s4b0n0:~> vllm --version
[W1106 16:55:04.833350173 OperatorEntry.cpp:218] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
    registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: XPU
  previous kernel: registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/aten/src/ATen/VmapModeRegistrations.cpp:37
       new kernel: registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/ipex_2.8.10_xpu_rel_08_18_2025/intel-extension-for-pytorch/build/Release/csrc/gpu/csrc/gpu/xpu/ATen/RegisterXPU_0.cpp:172 (function operator())
[2025-11-06 16:55:04,586] [INFO] [real_accelerator.py:260:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2025-11-06 16:55:06,888] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
INFO 11-06 16:55:07 [__init__.py:241] Automatically detected platform xpu.
0.10.1rc2.dev189+ge2db1164a.xpu
```

## Access Model Weights

Model weights for commonly used open-weight models are downloaded and available in the `/flare/datasets/model-weights/hub` directory on Aurora. To ensure your workflows utilize the preloaded model weights and datasets, update the following environment variables in your session. Some models hosted on Hugging Face may be gated, requiring additional authentication. To access these gated models, you will need a [Hugging Face authentication token](https://huggingface.co/docs/hub/en/security-tokens).
```bash linenums="1"
export HF_HOME="/flare/datasets/model-weights"
export HF_DATASETS_CACHE="/flare/datasets/model-weights"
export HF_MODULES_CACHE="/flare/datasets/model-weights"
export HF_TOKEN="YOUR_HF_TOKEN"
export RAY_TMPDIR="/tmp"
export TMPDIR="/tmp"
```

## Serve Small Models

For small models that fit within a single tile's memory (64 GB), no additional configuration is required to serve the model. Simply set `TP=1` (Tensor Parallelism). This configuration ensures the model is run on a single tile without the need for distributed setup. Models with fewer than 7 billion parameters typically fit within a single tile.

#### Using Single Tile

The following command serves `meta-llama/Llama-2-7b-chat-hf` on a single tile of a single node:
```bash linenums="1"
vllm serve meta-llama/Llama-2-7b-chat-hf
```

#### Using Multiple Tiles

To utilize multiple tiles for larger models (`TP>1`), a more advanced setup is necessary. First, configure a Ray cluster.
```bash linenums="1"
export VLLM_HOST_IP=$(getent hosts $(hostname).hsn.cm.aurora.alcf.anl.gov | awk '{ print $1 }' | tr ' ' '\n' | sort | head -n 1)
unset ONEAPI_DEVICE_SELECTOR # allow Ray to access all 12 GPU tiles
ray start --head --node-ip-address=$VLLM_HOST_IP --num-cpus=96 --num-gpus=12 &
export no_proxy="localhost,127.0.0.1" # Set no_proxy for the client to interact with the locally hosted model
```

The following script demonstrates how to serve the `meta-llama/Llama-2-7b-chat-hf` model across 8 tiles on a single node:

```bash linenums="1"
export VLLM_HOST_IP=$(getent hosts $(hostname).hsn.cm.aurora.alcf.anl.gov | awk '{ print $1 }' | tr ' ' '\n' | sort | head -n 1)
vllm serve meta-llama/Llama-2-7b-chat-hf --port 8000 --tensor-parallel-size 8 --trust-remote-code
```

## Serve Medium Models

#### Using Single Node

The following script demonstrates how to serve `meta-llama/Llama-3.3-70B-Instruct` on 8 tiles on a single node. Models with up to 70 billion parameters can usually fit within a single node, utilizing multiple tiles.

```bash linenums="1"
export VLLM_HOST_IP=$(getent hosts $(hostname).hsn.cm.aurora.alcf.anl.gov | awk '{ print $1 }' | tr ' ' '\n' | sort | head -n 1)
unset ONEAPI_DEVICE_SELECTOR # allow Ray to access all 12 GPU tiles
ray start --head --node-ip-address=$VLLM_HOST_IP --num-cpus=96 --num-gpus=12 &
vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 8 --trust-remote-code --max-model-len 32768
```

## Serve Large Models

### Using Multiple Nodes

coming soon...
