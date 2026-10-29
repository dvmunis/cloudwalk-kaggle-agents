// Normalize the webhook request: the competition slug is the only required input.
const body = $json.body ?? {};
if (!body.competition) {
  throw new Error('Body must include "competition" (Kaggle slug)');
}
const maxIterations = Math.round(Number(body.max_iterations ?? 3));
return [{
  json: {
    competition: body.competition,
    submit: body.submit === true,
    sample_frac: body.sample_frac ?? 1,
    max_iterations: Number.isFinite(maxIterations) ? Math.min(Math.max(maxIterations, 1), 10) : 3,
    goal: body.goal ?? 'Reach the top 20% of the public leaderboard',
  },
}];
