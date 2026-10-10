"""Hybrid MemE sharding; remote errors fall back to local inference."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import logging
import os
import urllib.request

from data_parallel import run_data_parallel


def request(url, path, data=None, headers=None):
    # This is direct server-to-server traffic, never through a user HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url.rstrip('/') + path, data=data, headers=headers or {})
    with opener.open(req, timeout=3600 if path == '/infer' else 120) as response:
        return json.load(response)


def remote_batch(url, items, bank_bytes, version):
    health = request(url, '/health')
    if health.get('bank_version') != version:
        synced = request(url, '/sync', gzip.compress(bank_bytes, compresslevel=1),
                         {'Content-Encoding': 'gzip', 'X-Bank-Version': version})
        if synced.get('bank_version') != version:
            raise RuntimeError('Remote MemE bank sync failed')
    result = request(url, '/infer', json.dumps({'trajectories': items, 'bank_version': version}).encode(),
                     {'Content-Type': 'application/json'})
    if result.get('bank_version') != version or len(result.get('results', [])) != len(items):
        raise RuntimeError('Remote MemE version/result count mismatch')
    return result['results']


def run_hybrid_meme_batch(items, *, gpu_ids, remote_urls, worker_fn, worker_kwargs):
    if not remote_urls:
        return run_data_parallel(items, gpu_ids, worker_fn, worker_kwargs)
    bank_bytes = json.dumps(worker_kwargs['bank'], ensure_ascii=False, separators=(',', ':')).encode()
    version = hashlib.sha256(bank_bytes).hexdigest()
    count = len(gpu_ids) + len(remote_urls)
    cap = max(1, int(os.getenv('REMOTE_MEME_ITEMS_PER_WORKER', '4')))
    shards = [[i for i in range(len(items)) if i % count == len(gpu_ids) + k]
              for k in range(len(remote_urls))]
    shards = [ids[:cap] for ids in shards]
    remote_ids = {i for ids in shards for i in ids}
    local_ids = [i for i in range(len(items)) if i not in remote_ids]
    merged = {}
    failed = []
    with ThreadPoolExecutor(max_workers=len(remote_urls)) as pool:
        jobs = [(ids, pool.submit(remote_batch, url, [items[i] for i in ids], bank_bytes, version))
                for url, ids in zip(remote_urls, shards) if ids]
        rows = run_data_parallel([items[i] for i in local_ids], gpu_ids, worker_fn, worker_kwargs)
        merged.update(zip(local_ids, rows))
        for ids, future in jobs:
            try:
                merged.update(zip(ids, future.result()))
            except Exception:
                logging.exception('Remote MemE failed; retry %s samples locally', len(ids))
                failed.extend(ids)
    if failed:
        rows = run_data_parallel([items[i] for i in failed], gpu_ids, worker_fn, worker_kwargs)
        merged.update(zip(failed, rows))
    logging.info('MemE hybrid finished: %s samples, %s remote-assigned, %s local fallbacks, bank=%s',
                 len(items), sum(map(len, shards)), len(failed), version)
    return [merged[i] for i in range(len(items))]
