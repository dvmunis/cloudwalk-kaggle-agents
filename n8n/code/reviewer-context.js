// What the LB Reviewer sees at the end of every iteration: the team's position on
// the public leaderboard (from its best submission so far), how far the goal is,
// what the team tried in this attempt and the plan it was following.
const input = $('Read input').first().json;
const it = $('Iteration context').first().json;
const run = $('Train').first().json;
let submission;
try {
  submission = $('Submit').isExecuted ? $('Submit').first()?.json : undefined;
} catch {
  submission = undefined;
}
const submitted = submission?.run_id === run.run_id;
const lb = $('Leaderboard').first().json;
const gap = lb.higher_is_better
  ? lb.top20_threshold - lb.best_public_score
  : lb.best_public_score - lb.top20_threshold;

return [{
  json: {
    attempt_id: it.attempt_id,
    run_id: run.run_id,
    context: {
      goal: input.goal,
      data_has_missing_values: Object.values($('Profile data').first().json.features).some((f) => f.null_pct > 0),
      iteration: `${it.iteration} of ${it.max_iterations}`,
      current_plan: it.plan,
      this_run: {
        run_id: run.run_id,
        cv_score: run.cv_score,
        cv_std: run.cv_std,
        submitted,
        public_score: submitted ? submission.public_score : null,
        model: run.config.model,
        params: run.config.params,
        features: run.config.features,
      },
      leaderboard: {
        rank: lb.rank,
        teams: lb.teams,
        top_percent: lb.top_percent,
        our_best_public_score: lb.best_public_score,
        top20_threshold: lb.top20_threshold,
        leader_score: lb.best_score,
        higher_is_better: lb.higher_is_better,
        gap_to_top20: Number(gap.toFixed(5)),
      },
      earlier_iterations: it.iterations,
      history: $('Refresh history').first().json.recent.filter((r) => r.sample_frac === input.sample_frac),
    },
  },
}];
