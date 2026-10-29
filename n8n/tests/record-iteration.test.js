const test = require('node:test');
const assert = require('node:assert/strict');
const { runNode } = require('./harness');

const plan = { strategy: ['A', 'B', 'C'], focus: 'B' };
const iteration = (iterations, extra = {}) => ({
  attempt_id: 'x', iteration: iterations.length + 1, max_iterations: 5, iterations, plan, ...extra,
});
const trained = { run_id: 'r1', cv_score: 0.9, cv_std: 0.001, higher_is_better: true,
  config: { model: 'lightgbm', params: {}, features: {} } };

function record(outputs) {
  return runNode('record-iteration.js', { 'Read input': { submit: true }, ...outputs });
}

test('a skipped iteration moves the focus to another plan step and explains why', () => {
  const out = record({
    'Iteration context': iteration([{ iteration: 1, run_id: 'r0' }]),
    'Build train config': { duplicate_of: 'r0', adjustments: ['configuration repeats run r0'] },
  });
  assert.equal(out.continue, true);
  assert.match(out.record.next_focus, /^A \(the previous proposal was not trained: configuration repeats run r0/);
});

test('two skips in a row end the attempt', () => {
  const out = record({
    'Iteration context': iteration([{ iteration: 1, skipped: 'repeat' }]),
    'Build train config': { duplicate_of: 'r0', adjustments: [] },
  });
  assert.equal(out.continue, false);
  assert.equal(out.stop_reason, 'agents repeated earlier runs twice in a row');
});

test('reaching the top 20% stops the loop', () => {
  const out = record({
    'Iteration context': iteration([]),
    'Build train config': { duplicate_of: null, adjustments: [] },
    Train: trained,
    Evaluator: { submit: true, verdict: 'improved', next_focus: 'n' },
    Submit: { run_id: 'r1', public_score: 0.95 },
    'Reviewer context': { run_id: 'r1' },
    'LB Reviewer': { assessment: 'ok', focus_for_next_iteration: 'f', stop: false },
    Leaderboard: { rank: 10, teams: 100, top_percent: 10, best_public_score: 0.95, top20_threshold: 0.94 },
  });
  assert.equal(out.record.submitted, true);
  assert.equal(out.continue, false);
  assert.equal(out.stop_reason, 'goal reached: top 10%');
});

test('outputs from an earlier iteration are not taken for this one', () => {
  const out = record({
    'Iteration context': iteration([], { max_iterations: 1 }),
    'Build train config': { duplicate_of: null, adjustments: [] },
    Train: trained,
    Evaluator: { submit: false, verdict: 'no_improvement', next_focus: 'try xgboost' },
    Submit: { run_id: 'older-run', public_score: 0.8 },
    'Reviewer context': { run_id: 'older-run' },
    'LB Reviewer': { focus_for_next_iteration: 'stale', stop: true },
  });
  assert.equal(out.record.submitted, false);
  assert.equal(out.record.next_focus, 'try xgboost');
  assert.equal(out.stop_reason, 'iteration budget used (1)');
});
