# GenMem PT and MT

This directory contains the LlamaFactory training package and GenMem's PT/MT workflow.

```text
src/        LlamaFactory source
pt/         SID alignment data, vocabulary expansion, and summary preparation
mt/         CoT generation, conversation assembly, and prompt alignment
configs/    GenMem PT, summary-continuation, MT, and DeepSpeed settings
```

The framework source is pinned to upstream [LlamaFactory](https://github.com/hiyouga/LlamaFactory) commit [`e67ab9e2f2c924a459b8586e3370b01dd5734b36`](https://github.com/hiyouga/LlamaFactory/commit/e67ab9e2f2c924a459b8586e3370b01dd5734b36), with its Apache-2.0 [LICENSE](LICENSE) preserved. `src/`, `pyproject.toml`, and `MANIFEST.in` are copied from that commit without changes. GenMem additions live in `pt/`, `mt/`, and `configs/`. Source hashes and provenance are recorded in `SOURCE_MANIFEST.json`.

Run all commands from the **GenMem repository root**, not this directory:

```bash
python -m pip install -r requirements-pt.txt
FORCE_TORCHRUN=1 llamafactory-cli train LlamaFactory/configs/pt.yaml
```

The requirements install this local source in editable mode. See the [environment guide](../docs/TRAINING.md), [summary preparation](pt/README.md), and [CoT-MT guide](mt/README.md) for data preparation and training. Configure your own input data and model paths before launching a run.
