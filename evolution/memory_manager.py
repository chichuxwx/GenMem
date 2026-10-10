"""
memory_manager.py

Handles memory bank loading, versioning (backup), and update.

Update strategy per iteration:
  - Collect (SID, evolved_experience, sp_score) from MemE outputs across all samples.
  - For each unique SID, pick the evolved_experience with highest sp_score.
  - Write it to `experience_summary` of the first matching entry in the bank.
  - If no entry exists at that SID, insert a new one.
"""

import copy
import json
import logging
import os
import re

logger = logging.getLogger(__name__)

_SID_RE = re.compile(
    r"<SID_L1_(\d+)><SID_L2_(\d+)><SID_L3_(\d+)><SID_L4_(\d+)>"
)


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------

def load_memory_bank(path: str) -> tuple[list[dict], dict]:
    """Load memory_bank.json. Returns (bank_list, sid_index).
    sid_index: (s1, s2, s3, s4) -> [entry, ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        bank = json.load(f)
    if isinstance(bank, dict):
        compact_bank = []
        for sid_text, summary in bank.items():
            match = _SID_RE.fullmatch(str(sid_text))
            if match is None or not str(summary).strip():
                continue
            s1, s2, s3, s4 = map(int, match.groups())
            compact_bank.append({
                "sample_id": f"summary_{s1}_{s2}_{s3}_{s4}",
                "dataset": "merged",
                "experience_type": "merged",
                "experience_text": str(summary),
                "experience_summary": str(summary),
                "sid": {
                    "s1": s1,
                    "s2": s2,
                    "s3": s3,
                    "s4": s4,
                    "tokens": str(sid_text),
                },
            })
        bank = compact_bank
    if not isinstance(bank, list):
        raise TypeError(f"Unsupported memory bank payload: {type(bank).__name__}")
    # Older merged-bank exports stored the fourth code in ``codes`` and in
    # the token string, but omitted ``sid.s4``.  Normalize that field on load
    # so exact four-level lookup and subsequent updates use the real SID.
    for entry in bank:
        sid = entry.get("sid")
        if not isinstance(sid, dict) or sid.get("s4") is not None:
            continue
        codes = sid.get("codes")
        if isinstance(codes, (list, tuple)) and len(codes) >= 4:
            sid["s4"] = int(codes[3])
            continue
        tokens = str(sid.get("tokens") or "")
        match = _SID_RE.search(tokens)
        if match:
            sid["s4"] = int(match.group(4))
    index = _build_index(bank)
    logger.info(f"Loaded {len(bank)} entries, {len(index)} unique SIDs from {path}")
    return bank, index


def save_memory_bank(bank: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bank, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(bank)} entries to {path}")


def _build_index(bank: list[dict]) -> dict:
    """(s1, s2, s3, s4) -> list[entry]."""
    index = {}
    for entry in bank:
        sid = entry.get("sid")
        if not sid:
            continue
        s4 = sid.get("s4")
        if s4 is None:
            codes = sid.get("codes")
            if isinstance(codes, (list, tuple)) and len(codes) >= 4:
                s4 = int(codes[3])
        key = (sid.get("s1"), sid.get("s2"), sid.get("s3"), s4)
        if None in key:
            continue
        index.setdefault(key, []).append(entry)
    return index


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------

def backup_memory(bank: list[dict], backup_dir: str, iter_idx: int) -> str:
    """Copy current bank to <backup_dir>/memory_iter_<iter_idx>/memory_bank.json.
    Does NOT modify bank.
    """
    out_dir = os.path.join(backup_dir, f"memory_iter_{iter_idx}")
    out_path = os.path.join(out_dir, "memory_bank.json")
    save_memory_bank(bank, out_path)
    logger.info(f"Backed up iter-{iter_idx} memory ({len(bank)} entries) → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------

def update_memory_bank(
    bank: list[dict],
    updates: list[dict],
    copy_bank: bool = True,
) -> tuple[list[dict], dict]:
    """
    Apply evolved experience updates to a COPY of the bank.

    Each update dict:
        sid_l1, sid_l2, sid_l3, sid_l4 : int
        evolved_experience      : str
        sp_score                : float   (quality signal; higher = better)
        question                : str     (for new-entry creation)
        dataset                 : str

    Strategy:
      1. For each unique SID, keep the update with highest sp_score.
      2. For matching entries in bank, overwrite experience_summary.
      3. If no entry exists at that SID, append a new one.

    Returns updated (new_bank, new_index).
    """
    # Step 1: deduplicate updates per SID (keep best sp_score)
    best: dict[tuple, dict] = {}
    for u in updates:
        if not u.get("evolved_experience"):
            continue
        key = (u["sid_l1"], u["sid_l2"], u["sid_l3"], u["sid_l4"])
        if key not in best or u.get("sp_score", 0.0) > best[key].get("sp_score", 0.0):
            best[key] = u

    logger.info(f"update_memory_bank: {len(updates)} raw updates → {len(best)} unique SIDs")

    # Step 2: deep-copy bank
    # The four-level production bank is about 8.7 GB on disk.  OfflineEvo does
    # not retain the previous in-memory version, so the runner can opt into an
    # in-place update and avoid duplicating the full object graph every round.
    new_bank = copy.deepcopy(bank) if copy_bank else bank
    new_index = _build_index(new_bank)

    updated_count = 0
    added_count = 0

    for (s1, s2, s3, s4), u in best.items():
        evolved = u["evolved_experience"].strip()
        if not evolved:
            continue

        entries = new_index.get((s1, s2, s3, s4), [])
        if entries:
            # Update the first (most representative) entry
            entries[0]["experience_summary"] = evolved
            updated_count += 1
        else:
            # Insert a new entry
            new_entry = {
                "sample_id": f"evolved_{s1}_{s2}_{s3}_{s4}",
                "dataset": u.get("dataset", "evolved"),
                "experience_type": "evolved",
                "experience_text": evolved,
                "experience_summary": evolved,
                "question": u.get("question", ""),
                "sid": {"s1": s1, "s2": s2, "s3": s3, "s4": s4,
                        "tokens": (
                            f"<SID_L1_{s1}><SID_L2_{s2}>"
                            f"<SID_L3_{s3}><SID_L4_{s4}>"
                        )},
            }
            new_bank.append(new_entry)
            new_index.setdefault((s1, s2, s3, s4), []).append(new_entry)
            added_count += 1

    logger.info(f"Memory update: {updated_count} updated, {added_count} added")
    return new_bank, new_index


# ---------------------------------------------------------------------------
# Lookup helper (mirrors MemoryTool logic, for offline use)
# ---------------------------------------------------------------------------

def lookup_sid(
    sid_str: str,
    bank: list[dict],
    index: dict,
    top_k: int = 3,
) -> str:
    """Given a SID token string, return formatted experience text."""
    m = _SID_RE.search(sid_str)
    if not m:
        return "[Memory lookup]: Invalid SID format."

    l1, l2, l3, l4 = map(int, m.groups())

    # Exact match
    entries = index.get((l1, l2, l3, l4), [])

    # L1+L2+L3 fallback
    if not entries:
        entries = [e for e in bank
                   if e.get("sid", {}).get("s1") == l1
                   and e.get("sid", {}).get("s2") == l2
                   and e.get("sid", {}).get("s3") == l3]

    # L1+L2 fallback
    if not entries:
        entries = [e for e in bank
                   if e.get("sid", {}).get("s1") == l1
                   and e.get("sid", {}).get("s2") == l2]

    # L1-only fallback
    if not entries:
        entries = [e for e in bank if e.get("sid", {}).get("s1") == l1]

    entries = entries[:top_k]
    if not entries:
        return f"[Memory lookup for {sid_str}]: No matching experience found."

    parts = []
    for idx, e in enumerate(entries):
        text = e.get("experience_summary") or e.get("experience_text", "")
        dataset = e.get("dataset", "?")
        parts.append(f"[{idx+1}] (Dataset: {dataset}) {text}")

    return f"[SID: {sid_str}]\n" + "\n".join(parts)
