// Start of an iteration (one training run). The plan is the Planner's until the
// LB Reviewer revises it; the focus comes from the previous iteration's feedback.
// Finished iterations are kept by "Record iteration" (its latest run has them all).
const input = $('Read input').first().json;
const planner = $('Planner').first().json;
const iterations = $('Record iteration').isExecuted ? $('Record iteration').first().json.iterations : [];
const last = iterations[iterations.length - 1];
const revised = $('LB Reviewer').isExecuted ? $('LB Reviewer').first().json : null;

return [{
  json: {
    attempt_id: $('Start attempt').first().json.attempt_id,
    iteration: iterations.length + 1,
    max_iterations: input.max_iterations,
    iterations,
    plan: {
      source: revised ? 'lb_reviewer' : 'planner',
      strategy: revised ? revised.plan : planner.strategy,
      focus: last ? last.next_focus : planner.focus_for_this_attempt,
    },
  },
}];
