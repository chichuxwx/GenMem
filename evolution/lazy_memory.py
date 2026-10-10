"""Replayable, batch-boundary memory commits stored with result checkpoints."""
import memory_manager as mm


def collect_updates(records, source):
    updates = []
    for rec in records:
        sid = rec.get(f"{source}_sid") or [None] * 4
        text = str(rec.get("evolved_experience") or "").strip()
        if (len(sid) != 4 or any(x is None for x in sid) or not text
                or rec.get("meme_source") == "fallback_failed"
                or "<think>" in text.lower()
                or "For multi-step arithmetic problems" in text
                or not rec.get("memory_update_gate_passed", False)):
            continue
        updates.append(dict(zip(("sid_l1", "sid_l2", "sid_l3", "sid_l4"), sid),
                            evolved_experience=text,
                            sp_score=1.0 if rec["sp_success"] else 0.5,
                            question=rec["question"], dataset=rec.get("dataset", "")))
    return updates


class LazyMemory:
    """The base is always memory_iter_(N-1); replay journal once on resume.

    Journal and results share ONE atomically replaced checkpoint. A crash
    before that replacement loses the uncommitted batch but never partially
    commits a bank. No full-bank copy is needed per flush.
    """
    def __init__(self, bank, index, state=None):
        self.bank, self.index = bank, index
        self.journal = list((state or {}).get("journal", []))
        self.applied = set()
        for commit in self.journal:
            ids = commit["indices"]
            if len(ids) != len(set(ids)) or self.applied.intersection(ids):
                raise ValueError("Duplicate sample in lazy memory journal")
            self.bank, self.index = mm.update_memory_bank(
                self.bank, commit["updates"], copy_bank=False)
            self.applied.update(ids)

    def flush(self, results, source, every, force=False):
        ids = [i for i, rec in enumerate(results)
               if rec is not None and i not in self.applied]
        if not ids or (not force and len(ids) < every):
            return False
        updates = collect_updates([results[i] for i in ids], source)
        self.bank, self.index = mm.update_memory_bank(self.bank, updates, copy_bank=False)
        self.journal.append({"indices": ids, "updates": updates})
        self.applied.update(ids)
        return True

    def state(self):
        return {"version": 1, "journal": self.journal}
