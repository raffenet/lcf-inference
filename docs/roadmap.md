# Roadmap

Planned development for Aegis.

## Frontier Support

- Add Slurm job generation and submission (alongside existing PBS support)
- Integrate ROCm/Apptainer-based vLLM launch workflow for AMD MI250X GPUs
- Validate weight staging on Frontier's Burst Buffer storage

## Additional Models

- Expand tested model list beyond Llama 3 and gpt-oss-120b
- Test and document multi-node configurations for models exceeding single-node GPU memory
- Add support for quantized model variants

## Larger Scale Testing

- Test beyond 1024 nodes on Aurora
- Benchmark instance startup time and weight staging throughput at scale
- Evaluate service registry performance under high instance counts

## Usability

- Improved error reporting and diagnostics for failed instances
- Support for custom vLLM server configurations beyond `extra_vllm_args`
- Configuration validation with actionable error messages
