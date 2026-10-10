"""Train the 48 x 16 x 8 x 8 weighted residual codebook from private embeddings."""

import argparse
import json
from pathlib import Path

import numpy as np

from .config import Config, DataConfig, ModelConfig, TrainConfig
from .rqkmeans_model_weighted import RQKMeansModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="JSONL from embedding/build_embeddings.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ids, vectors = [], []
    with args.input.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            ids.append(str(row["sample_id"]))
            vectors.append(row["embedding"])
    x = np.asarray(vectors, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 1024 or not np.isfinite(x).all():
        raise ValueError("Expected finite 1024-dimensional embeddings")
    if len(set(ids)) != len(ids):
        raise ValueError("sample_id must be unique")
    if not np.allclose(np.linalg.norm(x, axis=1), 1, atol=0.005):
        raise ValueError("Expected L2-normalized embeddings")
    sizes = [48, 16, 8, 8]
    model_config = ModelConfig(codebook_sizes=sizes, max_iter=20, nredo=3, random_state=args.seed)
    model_config.spherical_weight_per_level = [0.5] * 4
    cfg = Config(model=model_config, train=TrainConfig(str(args.output)), data=DataConfig(embedding_dim=1024))
    model = RQKMeansModel(cfg)
    model.fit(x)
    codes = model.encode(x)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    encoding_config = Path(__file__).resolve().parents[1] / "configs" / "config.json"
    (args.output / "config.json").write_text(encoding_config.read_text(encoding="utf-8"), encoding="utf-8")
    mapping = []
    for sample_id, row in zip(ids, codes):
        sid = "".join(f"<SID_L{i+1}_{int(v)}>" for i, v in enumerate(row))
        mapping.append({"sample_id": sample_id, "sid_tokens": sid})
    (args.output / "sid_mapping.json").write_text(json.dumps(mapping, ensure_ascii=False))


if __name__ == "__main__":
    main()
