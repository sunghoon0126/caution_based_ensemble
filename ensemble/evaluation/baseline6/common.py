"""Bounded JSONL grouping, provenance, and strict prefix resume."""

import hashlib
import json
import math
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError("Non-finite reward, loss or gradient")
    return value


def rows(path):
    with Path(path).open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL row: {path}:{number}")
            yield json.loads(line)


def groups(path, count):
    # Disk-backed unique index keeps memory bounded for millions of candidates.
    with tempfile.TemporaryDirectory(prefix="baseline6_ids_") as directory:
        with closing(sqlite3.connect(str(Path(directory) / "ids.sqlite"))) as ids:
            ids.execute("CREATE TABLE ids (id TEXT PRIMARY KEY)")
            yield from _groups(path, count, ids)


def _groups(path, count, ids):
    if count <= 0:
        raise ValueError("Candidate count must be positive")
    group, previous, experiment = [], None, None
    for row in rows(path):
        key = (row["dataset"], row["split"], int(row["trial_id"]),
               int(row["dataset_index"]), row["instance_id"])
        if experiment is not None and key[:3] != experiment:
            raise ValueError("Mixed dataset/split/trial")
        experiment = key[:3]
        if previous != key:
            if group:
                if len(group) != count:
                    raise ValueError("Incomplete candidate group")
                yield group
            if previous is not None and key[3] <= previous[3]:
                raise ValueError("Problems must be in increasing dataset_index order")
            group, previous = [], key
        if int(row["sample_index"]) != len(group) or len(group) >= count:
            raise ValueError("Unexpected sample_index or candidate count")
        if not row.get("candidate_id"):
            raise ValueError("Missing candidate_id")
        try:
            ids.execute("INSERT INTO ids VALUES (?)", (row["candidate_id"],))
        except sqlite3.IntegrityError as error:
            raise ValueError("Duplicate candidate_id") from error
        group.append(row)
    if group:
        if len(group) != count:
            raise ValueError("Incomplete candidate group")
        yield group


class ResumeWriter:
    """Validate every completed row against current input/config before skipping."""
    def __init__(self, path, config):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.config = config
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb") as check:
                check.seek(-1, 2)
                if check.read(1) != b"\n":
                    raise ValueError("Truncated output row; inspect/repair before resume")
        self.existing = iter(rows(path)) if self.path.exists() else iter(())
        self.appending = False
        self.dest = None

    def completed(self, source, context):
        fingerprint = digest({"source": source, "context": context})
        expected = {"input_fingerprint": fingerprint, "backprop_config": self.config}
        if not self.appending:
            prior = next(self.existing, None)
            if prior is not None:
                if any(prior.get(k) != v for k, v in expected.items()):
                    raise ValueError("Resume input/config mismatch; use a fresh output directory")
                if prior.get("candidate_id") != source["candidate_id"] or not prior.get("finite"):
                    raise ValueError("Invalid completed candidate")
                finite(prior["reward_after"])
                return True, expected
            self.appending = True
        return False, expected

    def write(self, row):
        if self.dest is None:
            self.dest = self.path.open("a", encoding="utf-8")
        self.dest.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self.dest.flush()

    def finish(self):
        if next(self.existing, None) is not None:
            raise ValueError("Output contains more candidates than input")

    def close(self):
        if self.dest:
            self.dest.close()
