#!/bin/bash -l

export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export no_proxy=localhost,127.0.0.1

module load frameworks

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Error: HF_TOKEN not set. Please export it and pass with qsub -v HF_TOKEN"
    exit 1
fi
export HF_HOME="/tmp/hf_home"
export VLLM_MODEL="meta-llama/Llama-3.3-70B-Instruct"

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=None
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NUMEXPR_MAX_THREADS=208

# output will be redirected to files with mpiexec
vllm serve $VLLM_MODEL --tensor-parallel-size 8

exit 0
