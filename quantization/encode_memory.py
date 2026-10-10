"""Encode memory text into a four-token SID; no MemR/MemE weights required."""
import argparse
import json
from pathlib import Path

import numpy as np


class SIDEncoder:
    def __init__(self, codebook_dir=None, load_codebooks=True):
        root = Path(codebook_dir) if codebook_dir else Path(__file__).parent / 'codebook'
        self.config = json.loads((root / 'config.json').read_text())
        self.codebooks = ([np.load(root / f'codebook_{i}.npy', allow_pickle=False)
                           for i in range(4)] if load_codebooks else [])
        for c, size in zip(self.codebooks, self.config['codebook_sizes']):
            if c.shape != (size, self.config['embedding_dim']) or not np.isfinite(c).all():
                raise ValueError('Invalid codebook')
        self.tokenizer = self.model = None

    def encode_embeddings(self, embeddings):
        """Accept already L2-normalized embedding vectors, NOT arbitrary model vectors."""
        if len(self.codebooks) != 4:
            raise ValueError('Load the four codebooks before assigning SIDs')
        residual = np.asarray(embeddings, dtype=np.float32)
        if residual.ndim == 1:
            residual = residual[None, :]
        if residual.ndim != 2 or residual.shape[1] != self.config['embedding_dim']:
            raise ValueError('Expected shape [N, 1024]')
        if not np.isfinite(residual).all():
            raise ValueError('Embeddings must be finite')
        if not np.allclose(np.linalg.norm(residual, axis=1), 1, atol=0.005):
            raise ValueError('Expected L2-normalized Qwen3 embeddings')
        codes = []
        for centers, weight in zip(self.codebooks, self.config['spherical_weight_per_level']):
            x2 = np.einsum('ij,ij->i', residual, residual)[:, None]
            c2 = np.einsum('ij,ij->i', centers, centers)[None, :]
            dot = residual @ centers.T
            euclidean = np.maximum(x2 + c2 - 2 * dot, 0)
            cosine_distance = np.maximum(1 - dot / (
                np.sqrt(np.maximum(x2, 1e-12)) * np.sqrt(np.maximum(c2, 1e-12))), 0)
            indices = np.argmin((1 - weight) * euclidean + weight * cosine_distance, axis=1)
            codes.append(indices)
            residual = residual - centers[indices]
        return np.stack(codes, axis=1)

    def load_embedding_model(self, device='cpu', model_path=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        cfg = self.config['embedding']
        kwargs = {} if model_path else {'revision': cfg['revision']}
        name = model_path or cfg['model']
        self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
        self.model = AutoModel.from_pretrained(name, torch_dtype=(
            torch.float16 if str(device).startswith('cuda') else torch.float32), **kwargs).to(device).eval()

    def embed(self, memories):
        import torch
        if self.model is None:
            self.load_embedding_model()
        result = []
        # Single-item encoding avoids the legacy padded-batch last-token ambiguity.
        for memory in memories:
            if not isinstance(memory, str) or not memory.strip():
                raise ValueError('Memory must be nonempty text')
            inputs = self.tokenizer(self.config['embedding']['instruction'] + memory,
                                    return_tensors='pt', truncation=True, max_length=8192)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state
                eos = torch.where(inputs['input_ids'][0] == self.tokenizer.eos_token_id)[0]
                index = int(eos[-1]) if len(eos) else hidden.shape[1] - 1
                vector = torch.nn.functional.normalize(hidden[:, index, :], p=2, dim=1)
            result.append(vector.float().cpu().numpy()[0])
        return np.asarray(result, dtype=np.float32)

    def encode(self, memories):
        if isinstance(memories, str):
            memories = [memories]
        if not memories:
            return []
        codes = self.encode_embeddings(self.embed(memories))
        return [{'sid_codes': row.tolist(), 'sid': ''.join(
            f'<SID_L{i+1}_{int(c)}>' for i, c in enumerate(row))} for row in codes]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--memory', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--embedding-model', help='Optional local Qwen3-Embedding-0.6B snapshot')
    args = p.parse_args()
    encoder = SIDEncoder()
    encoder.load_embedding_model(args.device, args.embedding_model)
    print(json.dumps(encoder.encode(args.memory)[0], ensure_ascii=False))
