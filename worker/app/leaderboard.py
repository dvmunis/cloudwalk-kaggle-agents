"""Where a score lands on the public leaderboard."""


def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def position(leaderboard_scores: list[float], score: float) -> dict:
    """Rank and top-percentile of `score` against a leaderboard ordered best-first.

    Direction is read from the leaderboard itself, so this works for any metric.
    """
    if not leaderboard_scores:
        raise ValueError("Leaderboard is empty")
    higher_is_better = leaderboard_scores[0] >= leaderboard_scores[-1]
    better = sum(1 for s in leaderboard_scores if (s > score if higher_is_better else s < score))
    rank = better + 1
    n = len(leaderboard_scores)
    return {
        "rank": rank,
        "teams": n,
        "top_percent": round(100 * rank / n, 2),
        "higher_is_better": higher_is_better,
        "best_score": leaderboard_scores[0],
        "top20_threshold": leaderboard_scores[max(int(n * 0.2) - 1, 0)],
    }


def summarize(rows: list[dict], own_submissions: list[dict]) -> dict:
    scores = [s for s in (_to_float(r.get("Score")) for r in rows) if s is not None]
    own = [s for s in (_to_float(r.get("publicScore")) for r in own_submissions) if s is not None]
    if not own:
        return {"teams": len(scores), "message": "No scored submissions yet"}
    higher_is_better = scores[0] >= scores[-1]
    best_own = max(own) if higher_is_better else min(own)
    return {"best_public_score": best_own, "n_submissions": len(own), **position(scores, best_own)}
