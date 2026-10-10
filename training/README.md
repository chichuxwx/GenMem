# Training environments

Run commands from the GenMem repository root. Use separate Python 3.12 environments for PT/MT and GRPO: their Transformers, PyTorch, and NumPy versions differ. The files below pin the core packages observed in the training environments, rather than mixing them with the inference requirements.

## PT and MT

```bash
python3.12 -m venv .venv-pt
source .venv-pt/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-pt.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip check
llamafactory-cli version
```

LlamaFactory is pinned to [`e67ab9e2f2c924a459b8586e3370b01dd5734b36`](https://github.com/hiyouga/LlamaFactory/commit/e67ab9e2f2c924a459b8586e3370b01dd5734b36). Its `src/` and package metadata match the training checkout; GenMem-specific data preparation and configurations live in this repository. Choose CUDA-compatible wheels and a FlashAttention build for your GPU. To use PyTorch attention instead, change `flash_attn` in the training YAML to `sdpa`.

Start with [summary preparation](pt/README.md) and [CoT mid-training](mt/README.md). Set the checkpoint and dataset paths in `configs/` before launching training.

## GRPO

```bash
python3.12 -m venv .venv-rl
source .venv-rl/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-rl.txt
python -m pip install flash-attn==2.8.3.post1 --no-build-isolation
python -m pip check
export VERL_ROOT="$PWD/third_party/verl"
```

The [vendored verl source](../third_party/verl/README.md) already contains the GenMem changes. Install it directly; there is no need to overlay a different upstream release. `training/rl/verl_overrides/` retains the focused GenMem changes for inspection.

Prepare RL parquet files with `training/rl/build_training_data.py` or `build_meme_training_data.py`. Set `MODEL_PATH` to the corresponding COT-MT checkpoint and provide absolute paths in `TRAIN_DATA_PATH`, `TEST_DATA_PATH`, `MEMORY_BANK_PATH`, and `SID_CONFIG_PATH`. Configure the outcome-scoring service with `DEERFLOW_BASE_URL`, `DEERFLOW_MODEL`, `DEERFLOW_EMAIL`, and `DEERFLOW_PASSWORD` in your environment. Train the task-solving model and configure its harness through [Harness-RL](https://github.com/jiangxinke/Harness-RL).

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash training/rl/run_memr.sh
# Or, with the MemE checkpoint and dataset paths:
bash training/rl/run_meme.sh
```

See `training/rl/config/local_config_v33.sh` for all environment settings. Launch one command at a time unless you assign disjoint resources. Before a full run, validate your own data and service with a short run, for example `TOTAL_TRAINING_STEPS=1 bash training/rl/run_memr.sh`.

## Local checks

```bash
python -m unittest discover -s tests -v
```

These CPU checks exercise data assembly and failure handling with synthetic data and mocked teacher responses. They do not launch GPU training or contact a teacher service.
