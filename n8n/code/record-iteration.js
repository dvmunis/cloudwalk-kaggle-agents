// End of an iteration: record what happened and decide whether to loop again.
// Nodes inside the loop keep the output of earlier iterations, so a node only
// counts for this iteration when its output refers to this iteration's run.
const it = $('Iteration context').first().json;
const built = $('Build train config').first().json;
const latest = (node, branch = 0) => {
  try {
    return $(node).isExecuted ? $(node).first(branch)?.json : undefined;
  } catch {
    return undefined;
  }
};

const finish = (record) => {
  let stopReason = null;
  if (record.top_percent <= 20) stopReason = `goal reached: top ${record.top_percent}%`;
  else if (record.reviewer_stop) stopReason = 'LB Reviewer stopped the attempt';
  else if (it.iteration >= it.max_iterations) stopReason = `iteration budget used (${it.max_iterations})`;
  return [{
    json: {
      attempt_id: it.attempt_id,
      record,
      iterations: [...it.iterations, record],
      continue: stopReason === null,
      stop_reason: stopReason,
    },
  }];
};

// The guardrails skipped training: the proposal (after corrections) repeated an
// earlier run. Agents tend to propose the same thing again, so the focus moves to
// another step of the plan, and two skips in a row end the attempt.
if (built.duplicate_of) {
  let skipsInRow = 1;
  for (let i = it.iterations.length - 1; i >= 0 && it.iterations[i].skipped; i--) skipsInRow++;
  const otherSteps = (it.plan.strategy ?? []).filter((step) => step !== it.plan.focus);
  const nextStep = otherSteps[(skipsInRow - 1) % Math.max(otherSteps.length, 1)];
  const record = {
    iteration: it.iteration,
    focus: it.plan.focus,
    skipped: `configuration repeats run ${built.duplicate_of}`,
    guardrail_adjustments: built.adjustments,
    next_focus: `${nextStep ?? 'A change not tried yet'} (the previous proposal was not trained: `
      + `${built.adjustments.join('; ')}. Propose a configuration that is not in the history.)`,
  };
  const result = finish(record);
  if (skipsInRow >= 2 && result[0].json.continue) {
    Object.assign(result[0].json, { continue: false, stop_reason: 'agents repeated earlier runs twice in a row' });
  }
  return result;
}

const train = $('Train').first().json;
const evaluation = $('Evaluator').first().json;
const submission = latest('Submit');
const submitted = submission?.run_id === train.run_id;
const wantedToSubmit = $('Read input').first().json.submit && evaluation.submit === true;
const reviewed = latest('Reviewer context')?.run_id === train.run_id;
const reviewer = reviewed ? $('LB Reviewer').first().json : null;
const lb = reviewed ? latest('Leaderboard') : undefined;

const record = {
  iteration: it.iteration,
  focus: it.plan.focus,
  run_id: train.run_id,
  cv_score: train.cv_score,
  cv_std: train.cv_std,
  higher_is_better: train.higher_is_better,
  model: train.config.model,
  params: train.config.params,
  features: train.config.features,
  guardrail_adjustments: built.adjustments,
  verdict: evaluation.verdict,
  submitted,
  next_focus: reviewer ? reviewer.focus_for_next_iteration : evaluation.next_focus,
};
if (reviewer) record.lb_assessment = reviewer.assessment;
if (reviewer?.stop === true) record.reviewer_stop = true;
if (submitted) record.public_score = submission.public_score;
if (lb?.rank) {
  Object.assign(record, {
    rank: lb.rank,
    teams: lb.teams,
    top_percent: lb.top_percent,
    best_public_score: lb.best_public_score,
    top20_threshold: lb.top20_threshold,
  });
}
if (wantedToSubmit && !submitted) {
  const error = latest('Submit', 1);
  record.submit_error = error?.error?.message ?? error?.error ?? 'submission did not complete';
}

return finish(record);
