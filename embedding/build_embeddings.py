"""Encode experience text with the same prompt and pooling used by the SID codebook."""

import argparse
import json
from pathlib import Path

import numpy as np

from quantization.encode_memory import SIDEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="JSONL records with sample_id and experience_text")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--embedding-model", default=None)
    args = parser.parse_args()

    encoder = SIDEncoder(Path(__file__).resolve().parents[1] / "configs", load_codebooks=False)
    encoder.load_embedding_model(args.device, args.embedding_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", "")).strip()
            memory = str(row.get("experience_text", "")).strip()
            if not sample_id or not memory:
                raise ValueError(f"Missing sample_id or experience_text at line {line_number}")
            vector = encoder.embed([memory])[0]
            assert vector.shape == (1024,) and np.isfinite(vector).all()
            target.write(json.dumps({"sample_id": sample_id, "embedding": vector.tolist()}) + "\n")


if __name__ == "__main__":
    main()
