# verl for GenMem

This is a source-only snapshot of the verl package used by GenMem, based on the training repository commit `f98e27e44a11aaf25ade0b96e5a32320ed30cc22` (subdirectory `verl/`). This identifier records local provenance; install this directory directly rather than fetching that commit from upstream.

Upstream: [verl](https://github.com/verl-project/verl). The upstream Apache-2.0 [LICENSE](LICENSE) and [Notice.txt](Notice.txt) are preserved.

The snapshot contains the tracked `verl/` Python package, configuration files, build metadata, and GenMem's single-lookup recipe. Training outputs and neuron-analysis artifacts were excluded. No model weights or memory-bank payloads are needed to install it.

GenMem changes include agent/tool parsing, one-lookup tool loops, process-reward handling, actor losses and training integration, four-layer memory lookup, and vLLM rollout handling. The focused modified files are mirrored in [`training/rl/verl_overrides/`](../../training/rl/verl_overrides/). Machine-specific defaults in the legacy memory tool and two legacy StackPlanner reward modules were replaced with relative paths or `STACK_PLANNER_ROOT` / `STACK_PLANNER_PYTHON` environment settings. The GenMem launchers select their own four-layer tool and reward modules.

Use the [GRPO environment instructions](../../training/README.md) from the GenMem root:

```bash
python -m pip install -r requirements-rl.txt
export VERL_ROOT="$PWD/third_party/verl"
```

Source checksums are recorded in `SOURCE_MANIFEST.json` for inspection. Use this snapshot as a unit: replacing it with a current upstream checkout can change rollout, tool-call, or reward behavior.
