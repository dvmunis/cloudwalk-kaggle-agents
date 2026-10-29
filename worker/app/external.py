"""External data: public Kaggle datasets with the same columns as the competition.

Playground competitions are generated from a real "original" dataset, and adding
its rows to training usually helps. Discovery searches Kaggle, downloads small
candidates and accepts only files that have the competition's target and most of
its features. Accepted datasets are stored normalized to the competition's
columns; whether to train with them is the agents' decision.
"""

import hashlib
import json
import re
import shutil
from pathlib import Path

import pandas as pd

from . import kaggle_client, settings
from .profiling import CompetitionData

MIN_FEATURE_MATCH = 0.8
MAX_DOWNLOAD_MB = 100
CANDIDATES_PER_QUERY = 5
MAX_DOWNLOADS = 8


def _dir(slug: str) -> Path:
    return settings.competition_data_dir(slug) / "external"


def _registry_path(slug: str) -> Path:
    return _dir(slug) / "registry.json"


def _registry(slug: str) -> dict:
    path = _registry_path(slug)
    return json.loads(path.read_text()) if path.exists() else {}


def _save_registry(slug: str, registry: dict) -> None:
    _dir(slug).mkdir(parents=True, exist_ok=True)
    _registry_path(slug).write_text(json.dumps(registry, indent=2))


def default_queries(data: CompetitionData) -> list[str]:
    """A search built from the target's name, e.g. Will_Buy_EV -> 'will buy ev'."""
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", data.target).replace("_", " ").lower().split()
    return [" ".join(words)] if words else []


def _is_competition_derivative(ref: str, slug: str) -> bool:
    """Datasets built from this competition's own files (copies, OOF predictions, features)."""
    name = ref.lower()
    tokens = [t for t in slug.lower().split("-") if any(ch.isdigit() for ch in t) and len(t) >= 3]
    return slug.lower() in name or "playground" in name or any(t in name for t in tokens)


def _match_columns(columns: list[str], data: CompetitionData) -> dict[str, str]:
    """File column -> competition column, by case- and separator-insensitive name."""
    key = lambda c: re.sub(r"[^a-z0-9]", "", str(c).lower())  # noqa: E731
    wanted = {key(c): c for c in [*data.feature_cols, data.target]}
    return {c: wanted[key(c)] for c in columns if key(c) in wanted}


def _inspect(path: Path, data: CompetitionData) -> tuple[pd.DataFrame | None, str]:
    """Normalized frame if the file can be used as extra training rows, else the reason it cannot."""
    try:
        header = pd.read_csv(path, nrows=5)
    except Exception as exc:  # noqa: BLE001 - any unreadable file is just rejected
        return None, f"unreadable: {exc}"
    mapping = _match_columns(list(header.columns), data)
    if data.target not in mapping.values():
        return None, f"no '{data.target}' column"
    matched = [c for c in mapping.values() if c != data.target]
    share = len(matched) / len(data.feature_cols)
    if share < MIN_FEATURE_MATCH:
        return None, f"only {len(matched)} of {len(data.feature_cols)} feature columns"

    frame = pd.read_csv(path, usecols=list(mapping)).rename(columns=mapping)
    if len(frame) in (len(data.train), len(data.test)):
        return None, "same number of rows as the competition files (likely a copy)"
    frame = frame.dropna(subset=[data.target])
    if data.task == "regression":
        frame[data.target] = pd.to_numeric(frame[data.target], errors="coerce")
        frame = frame.dropna(subset=[data.target])
    else:
        classes = {str(c) for c in data.classes}
        unknown = set(frame[data.target].astype(str)) - classes
        if unknown:
            return None, f"target values {sorted(unknown)[:5]} are not competition classes"
        lookup = {str(c): c for c in data.classes}
        frame[data.target] = frame[data.target].astype(str).map(lookup)
    if frame.empty:
        return None, "no usable rows"
    for col in data.feature_cols:
        if col not in frame:
            frame[col] = pd.NA
    return frame[[*data.feature_cols, data.target]], f"{len(matched)} of {len(data.feature_cols)} features"


def discover(data: CompetitionData, queries: list[str]) -> dict:
    """Search Kaggle and register datasets that can serve as extra training rows."""
    registry = _registry(data.slug)
    queries = list(dict.fromkeys([q.strip() for q in [*queries, *default_queries(data)] if q and q.strip()]))
    candidates, searched = [], []
    for query in queries:
        rows = kaggle_client.search_datasets(query)
        searched.append({"query": query, "results": len(rows)})
        for row in rows[:CANDIDATES_PER_QUERY]:
            ref = row.get("ref", "")
            if ref and ref not in registry and ref not in [c["ref"] for c in candidates]:
                candidates.append(row)

    for row in candidates[:MAX_DOWNLOADS]:
        ref = row["ref"]
        size_mb = float(row.get("size") or 0) / 1e6
        entry = {"ref": ref, "title": row.get("title", "")}
        if _is_competition_derivative(ref, data.slug):
            registry[ref] = {**entry, "status": "rejected", "reason": "built from this competition's files"}
            continue
        if size_mb > MAX_DOWNLOAD_MB:
            registry[ref] = {**entry, "status": "rejected", "reason": f"too large ({size_mb:.0f} MB)"}
            continue
        download_dir = _dir(data.slug) / "_downloads" / ref.replace("/", "__")
        try:
            files = kaggle_client.download_dataset(ref, download_dir)
        except kaggle_client.KaggleError as exc:
            registry[ref] = {**entry, "status": "rejected", "reason": f"download failed: {exc}"}
            shutil.rmtree(download_dir, ignore_errors=True)
            continue
        results = [(path, *_inspect(path, data)) for path in files]
        usable = [(path, frame, note) for path, frame, note in results if frame is not None]
        if not usable:
            reasons = "; ".join(f"{p.name}: {note}" for p, _, note in results) or "no CSV files"
            registry[ref] = {**entry, "status": "rejected", "reason": reasons}
        else:
            path, frame, note = max(usable, key=lambda item: len(item[1]))
            # Re-uploads of the same data would count the same rows twice.
            digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()
            twin = next((r for r, e in registry.items() if e.get("sha256") == digest and e["status"] == "accepted"), None)
            if twin:
                registry[ref] = {**entry, "status": "rejected", "reason": f"same rows as {twin}"}
            else:
                name = ref.replace("/", "__")
                frame.to_csv(_dir(data.slug) / f"{name}.csv", index=False)
                registry[ref] = {**entry, "status": "accepted", "name": name, "file": path.name, "rows": len(frame),
                                 "columns": note, "sha256": digest,
                                 "missing_columns": [c for c in data.feature_cols if frame[c].isna().all()]}
        shutil.rmtree(download_dir, ignore_errors=True)

    _save_registry(data.slug, registry)
    return {"searched": searched, "accepted": available(data.slug),
            "rejected": [{"ref": r, "reason": e["reason"]} for r, e in registry.items() if e["status"] == "rejected"]}


def available(slug: str) -> list[dict]:
    """Accepted datasets, as the agents see them."""
    return [{k: e[k] for k in ("name", "ref", "title", "rows", "missing_columns")}
            for e in _registry(slug).values() if e["status"] == "accepted"]


def load(data: CompetitionData, names: list[str]) -> pd.DataFrame | None:
    """Accepted datasets as extra training rows, with the dtypes of the competition's train file."""
    known = {e["name"] for e in available(data.slug)}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"external_data: unknown datasets {unknown}; available: {sorted(known)}")
    if not names:
        return None
    frame = pd.concat([pd.read_csv(_dir(data.slug) / f"{n}.csv") for n in names], ignore_index=True)
    for col in data.feature_cols:
        if pd.api.types.is_numeric_dtype(data.train[col]):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        else:
            frame[col] = frame[col].astype(object).where(frame[col].notna(), None)
    return frame
