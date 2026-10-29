"""Thin wrapper over the Kaggle CLI.

The CLI is used instead of the Python API because its interface has been more
stable across package versions. Credentials come from KAGGLE_USERNAME /
KAGGLE_KEY or ~/.kaggle/kaggle.json, as the CLI expects.
"""

import csv
import io
import subprocess
import time
import zipfile
from pathlib import Path


class KaggleError(RuntimeError):
    pass


def _run(args: list[str]) -> str:
    proc = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise KaggleError((proc.stderr or proc.stdout).strip())
    return proc.stdout


def _parse_csv(output: str, header_marker: str) -> list[dict]:
    # The CLI sometimes prints version warnings before the CSV header.
    lines = output.splitlines()
    start = next((i for i, line in enumerate(lines) if header_marker in line), None)
    if start is None:
        return []
    return list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))


def _extract_zips(directory: Path) -> None:
    for archive in directory.glob("*.zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(directory)
        archive.unlink()


def download(slug: str, dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    _run(["competitions", "download", "-c", slug, "-p", str(dest)])
    _extract_zips(dest)
    return sorted(p.name for p in dest.iterdir())


def search_datasets(query: str) -> list[dict]:
    """Public datasets matching a search: ref, title, size (bytes), lastUpdated, ..."""
    return _parse_csv(_run(["datasets", "list", "-s", query, "--csv"]), "ref")


def download_dataset(ref: str, dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    _run(["datasets", "download", "-d", ref, "-p", str(dest), "--unzip", "-q"])
    _extract_zips(dest)
    return sorted(dest.rglob("*.csv"))


def submit(slug: str, file: Path, message: str) -> None:
    _run(["competitions", "submit", "-c", slug, "-f", str(file), "-m", message])


def submissions(slug: str) -> list[dict]:
    """Own submissions, newest first: fileName, date, description, status, publicScore, privateScore."""
    return _parse_csv(_run(["competitions", "submissions", "-c", slug, "--csv"]), "publicScore")


def wait_for_score(slug: str, description: str, timeout_s: int = 300, poll_s: int = 10) -> dict:
    """Poll until the submission with this description has a public score."""
    deadline = time.monotonic() + timeout_s
    while True:
        match = next((s for s in submissions(slug) if s.get("description") == description), None)
        status = (match or {}).get("status", "").lower()
        if match and "error" in status:
            raise KaggleError(f"Submission failed on Kaggle: {match}")
        if match and "complete" in status and match.get("publicScore") not in ("", "None", None):
            return match
        if time.monotonic() > deadline:
            raise KaggleError(f"Timed out waiting for score of submission '{description}'")
        time.sleep(poll_s)


def leaderboard(slug: str, dest: Path) -> list[dict]:
    """Full public leaderboard, ordered by rank."""
    dest.mkdir(parents=True, exist_ok=True)
    _run(["competitions", "leaderboard", "-c", slug, "--download", "-p", str(dest)])
    _extract_zips(dest)
    latest = max(dest.glob("*.csv"), key=lambda p: p.stat().st_mtime, default=None)
    if latest is None:
        raise KaggleError("Leaderboard download produced no CSV")
    with latest.open() as f:
        return list(csv.DictReader(f))
