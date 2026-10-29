// Outcome of the attempt: every iteration, the best run and the latest leaderboard position.
const loop = $('Record iteration').first().json;
const iterations = loop.iterations;
const higherIsBetter = iterations.find((i) => i.run_id)?.higher_is_better ?? true;
const best = iterations.filter((i) => i.run_id).reduce((a, b) => {
  if (!a) return b;
  return (higherIsBetter ? b.cv_score > a.cv_score : b.cv_score < a.cv_score) ? b : a;
}, null);

const summary = {
  iterations: iterations.length,
  stop_reason: loop.stop_reason,
  best_run: best && { iteration: best.iteration, run_id: best.run_id, cv_score: best.cv_score, model: best.model },
  submissions: iterations.filter((i) => i.submitted).length,
  history: iterations.map((i) => ({
    iteration: i.iteration,
    cv_score: i.cv_score,
    verdict: i.verdict ?? i.skipped,
    public_score: i.public_score,
    top_percent: i.top_percent,
    guardrail_adjustments: i.guardrail_adjustments.length,
  })),
};
const lastReviewed = [...iterations].reverse().find((i) => i.rank);
if (lastReviewed) {
  const { best_public_score, rank, teams, top_percent, top20_threshold } = lastReviewed;
  Object.assign(summary, { best_public_score, rank, teams, top_percent, top20_threshold });
}
return [{ json: { summary } }];
