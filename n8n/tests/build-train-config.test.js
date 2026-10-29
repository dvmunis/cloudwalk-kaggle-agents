const test = require('node:test');
const assert = require('node:assert/strict');
const { runNode } = require('./harness');

const FEATURES = { drop: [], frequency_encode: [], as_category: [], target_encode: [], digit_features: [],
  add_null_count: false };
const profile = {
  suggested_metric: 'roc_auc',
  features: { age: { null_pct: 0, min: 18 }, income: { null_pct: 0, min: 0 }, city: { null_pct: 0 } },
};
const run = (id, extra) => ({
  run_id: id, model: 'lightgbm', params: {}, features: FEATURES, external_data: [], sample_frac: 1,
  blendable: true, cv_score: 0.9, ...extra,
});

function guard({ data = {}, modeling = {}, runs = [], best, accepted = [], metric = 'roc_auc' } = {}) {
  return runNode('build-train-config.js', {
    'Profile data': profile,
    Planner: { metric },
    'Iteration context': { iteration: 1, plan: { focus: 'test' } },
    'Refresh history': { best_full_data_run: best ?? null, best_full_data_model_run: best ?? null, recent: runs },
    'Data agent': { ...FEATURES, ...data },
    'Modeling agent': { model: 'lightgbm', params: {}, blend_runs: [], ...modeling },
    'Read input': { sample_frac: 1 },
    'Discover external data': { accepted },
  });
}

test('unknown columns, metrics and params are corrected, not trusted', () => {
  const out = guard({
    data: { frequency_encode: ['income', 'ghost'] },
    modeling: { params: { num_leaves: 9999, made_up: 1, learning_rate: 0.05 } },
    metric: 'accuracy_top1',
  });
  assert.equal(out.config.metric, 'roc_auc');
  assert.deepEqual(out.config.features.frequency_encode, ['income']);
  assert.deepEqual(out.config.params, { num_leaves: 512, learning_rate: 0.05 });
  assert.equal(out.adjustments.length, 4);
});

test('a null count is dropped when the data has no missing values', () => {
  const out = guard({ data: { add_null_count: true } });
  assert.equal(out.config.features.add_null_count, false);
  assert.match(out.adjustments[0], /no missing values/);
});

test('digit features are kept only for numeric columns', () => {
  const out = guard({ data: { digit_features: ['income', 'city'] } });
  assert.deepEqual(out.config.features.digit_features, ['income']);
});

test('a feature change resets params to the reference run, so its effect can be attributed', () => {
  const best = run('best', { params: { num_leaves: 127 } });
  const out = guard({ best, runs: [best], data: { target_encode: ['city'] }, modeling: { params: { num_leaves: 255 } } });
  assert.deepEqual(out.config.params, { num_leaves: 127 });
  assert.match(out.adjustments[0], /features changed vs reference run best/);
});

test('at most two hyperparameters change against the reference run', () => {
  const best = run('best', { params: { num_leaves: 127 } });
  const out = guard({ best, runs: [best], modeling: { params: { num_leaves: 255, learning_rate: 0.03, subsample: 0.9 } } });
  assert.deepEqual(out.config.params, { num_leaves: 255, learning_rate: 0.03 });
  assert.match(out.adjustments[0], /'subsample' reverted/);
});

test('the reference is a run with the same features when one exists', () => {
  const best = run('best', { params: { num_leaves: 127 }, cv_score: 0.95 });
  const encoded = run('encoded', { features: { ...FEATURES, target_encode: ['city'] }, params: { num_leaves: 127 } });
  const out = guard({ best, runs: [best, encoded], data: { target_encode: ['city'] }, modeling: { params: { num_leaves: 255 } } });
  assert.deepEqual(out.config.params, { num_leaves: 255 });
  assert.equal(out.duplicate_of, null);
});

test('a configuration already trained is skipped', () => {
  const best = run('best', { params: { num_leaves: 127 } });
  const out = guard({ best, runs: [best], modeling: { params: { num_leaves: 127 } } });
  assert.equal(out.duplicate_of, 'best');
});

test('repeating a run without saved predictions is allowed, so it can be blended', () => {
  const old = run('old', { blendable: false });
  const out = guard({ best: old, runs: [old] });
  assert.equal(out.duplicate_of, null);
  assert.match(out.adjustments[0], /no saved predictions; retrained/);
});

test('blends use only blendable runs and retrain a member that lacks predictions', () => {
  const a = run('a');
  const b = run('b', { model: 'xgboost' });
  const old = run('old', { blendable: false, params: { num_leaves: 31 } });
  const ok = guard({ best: a, runs: [a, b, old], modeling: { model: 'blend', blend_runs: ['a', 'b', 'a'] } });
  assert.deepEqual(ok.config.blend_runs, ['a', 'b']);

  const retrain = guard({ best: a, runs: [a, old], modeling: { model: 'blend', blend_runs: ['a', 'old'] } });
  assert.equal(retrain.config.model, 'lightgbm');
  assert.deepEqual(retrain.config.params, { num_leaves: 31 });
  assert.match(retrain.adjustments.at(-1), /retraining run old as it was/);
});

test('a blend already tried is skipped', () => {
  const a = run('a');
  const b = run('b');
  const done = run('blend1', { model: 'blend', blend_runs: ['b', 'a'] });
  const out = guard({ best: a, runs: [a, b, done], modeling: { model: 'blend', blend_runs: ['a', 'b'] } });
  assert.equal(out.duplicate_of, 'blend1');
});

test('external data is limited to accepted datasets and counts as a feature change', () => {
  const best = run('best', { params: { num_leaves: 127 } });
  const out = guard({
    best, runs: [best], accepted: [{ name: 'orig' }],
    data: { external_data: ['orig', 'random'] }, modeling: { params: { num_leaves: 255 } },
  });
  assert.deepEqual(out.config.external_data, ['orig']);
  assert.deepEqual(out.config.params, { num_leaves: 127 });
  assert.ok(out.adjustments.some((a) => a.includes('unknown dataset random')));
});
