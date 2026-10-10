# GenMem

Code for semantic-ID memory retrieval and evolution. Model weights and the four-layer codebook are available on [Hugging Face](https://huggingface.co/chuchuxwx/GenMem).

![GenMem method](assets/method.png)

[Figure PDF](assets/GenMen_method%284%29%281%29.pdf)

## Quick start

```bash
pip install -r requirements.txt
hf download chuchuxwx/GenMem --include 'codebook/*' --local-dir quantization
python -m quantization.encode_memory --memory 'Your reusable experience' --device cuda
```

This maps a memory text to a four-token SID. The released codebook has sizes **48 × 16 × 8 × 8**. The embedding model is downloaded automatically on first use.

## Train from your own data

The main entry points are:

| Stage | Code |
| --- | --- |
| Experience embeddings | [`embedding/build_embeddings.py`](embedding/build_embeddings.py) |
| RQ-KMeans codebook | [`quantization/train_rqk.py`](quantization/train_rqk.py) |
| SID pretraining | [`training/pt/`](training/pt/) and [`configs/pt.yaml`](configs/pt.yaml) |
| MemR / MemE CoT mid-training | [`training/mt/`](training/mt/) and [`configs/`](configs/) |
| MemR / MemE GRPO | [`training/rl/`](training/rl/) |
| Offline memory evolution | [`evolution/offline_evo.py`](evolution/offline_evo.py) |

For example, to build a new codebook from a private JSONL file with `sample_id` and `experience_text` fields:

```bash
python -m embedding.build_embeddings --input experiences.jsonl --output embeddings.jsonl --device cuda
python -m quantization.train_rqk --input embeddings.jsonl --output codebook
```

PT and MT use [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory). GRPO uses a compatible [verl](https://github.com/volcengine/verl) checkout with the files in `training/rl/verl_overrides/` applied. Set the model, dataset, memory-bank, and solver paths in the configs before running them. The offline evolution CLI lists its options with `python evolution/offline_evo.py --help`.

Training data, memory banks, trajectories, and solver services are not included.

## License

Apache-2.0. Third-party model and dependency licenses apply separately.
