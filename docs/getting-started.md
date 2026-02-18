# Getting Started

## Prerequisites

- Python 3.9+
- Access to an HPC cluster with PBS (Aurora) or Slurm (Frontier, planned)
- vLLM (can be installed as part of a conda environment staged by Aegis via the `--conda-env` option, or available on compute nodes via `module load frameworks` on Aurora)

## Installation

Install from the repository root:

```bash
pip install .
```

For development (editable install):

```bash
pip install -e .
```

Verify the installation:

```bash
aegis --help
```

## First Launch

### 1. Create a config file

Create a file called `config.yaml`:

```yaml
model: meta-llama/Llama-3.3-70B-Instruct
instances: 2
tensor_parallel_size: 6
port_start: 8000
hf_home: /tmp/hf_home
model_source: /flare/datasets/model-weights/hub/models--meta-llama--Llama-3.3-70B-Instruct
walltime: "01:00:00"
account: MyProject
filesystems: flare:home
extra_vllm_args:
  - --max-model-len
  - "32768"
```

### 2. Preview the generated PBS script

Use `--dry-run` to see what would be submitted without actually submitting:

```bash
aegis submit --config config.yaml --dry-run
```

### 3. Submit the job

```bash
aegis submit --config config.yaml
```

This generates a PBS batch script and submits it via `qsub`. The job will call `aegis launch` inside the allocation to stage weights and start vLLM instances.

### 4. Launch inside an existing allocation

If you already have a PBS allocation (e.g., via `qsub -I`), launch instances directly:

```bash
aegis launch --config config.yaml
```

### 5. Query running instances

Once instances are running, use the service registry to discover them:

```bash
aegis registry list
aegis registry list-healthy
```

## Overriding Config via CLI

All config values can be overridden with CLI flags. CLI flags take precedence over the config file:

```bash
aegis submit \
    --config config.yaml \
    --instances 4 \
    --walltime 02:00:00 \
    --dry-run
```

See [CLI Reference](cli.md) for all available flags and [Configuration](configuration.md) for the full config format.
