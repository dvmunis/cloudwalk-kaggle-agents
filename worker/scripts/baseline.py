"""No-LLM baseline: download, profile, train and (optionally) submit one competition.

It exercises the same functions the API exposes, proving the execution layer
works before any agent is involved.

    python -m scripts.baseline --competition playground-series-s5e10 --sample-frac 0.2
    python -m scripts.baseline --competition playground-series-s5e10 --submit
"""

import argparse
import json

from app import kaggle_client, leaderboard, profiling, settings, training
from app.schemas import SubmitRequest, TrainConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--competition", required=True, help="Kaggle competition slug")
    parser.add_argument("--metric", help="Override the inferred metric (check the competition's Evaluation page)")
    parser.add_argument("--model", default="lightgbm", choices=["lightgbm", "xgboost"])
    parser.add_argument("--sample-frac", type=float, default=1.0)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    slug = args.competition
    if not args.skip_download:
        print("Downloading:", kaggle_client.download(slug, settings.competition_data_dir(slug)))

    data = profiling.load(slug)
    summary = profiling.profile(data)
    print(json.dumps({k: v for k, v in summary.items() if k != "features"}, indent=2))

    metric = args.metric or summary["suggested_metric"]
    if not args.metric:
        print(f"Using inferred metric '{metric}'. Confirm it on the competition's Evaluation page.")

    cfg = TrainConfig(metric=metric, model=args.model, sample_frac=args.sample_frac, note="baseline")
    result = training.train(data, cfg)
    print(f"Run {result['run_id']}: CV {result['metric']} = {result['cv_score']:.5f} "
          f"(± {result['cv_std']:.5f}) in {result['duration_s']}s")

    if args.submit:
        req = SubmitRequest(run_id=result["run_id"], message="baseline")
        file = settings.competition_runs_dir(slug) / req.run_id / "submission.csv"
        description = f"{req.run_id} {req.message}"
        kaggle_client.submit(slug, file, description)
        scored = kaggle_client.wait_for_score(slug, description)
        print("Public score:", scored["publicScore"])
        rows = kaggle_client.leaderboard(slug, settings.competition_runs_dir(slug) / "_leaderboard")
        print(json.dumps(leaderboard.summarize(rows, kaggle_client.submissions(slug)), indent=2))


if __name__ == "__main__":
    main()
