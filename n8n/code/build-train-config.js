// Guardrails between the agents and the worker: agent output is checked against
// the allowed options and corrected rather than trusted. Every correction is
// reported so it shows up in the trace.
const profile = $('Profile data').first().json;
const planner = $('Planner').first().json;
const it = $('Iteration context').first().json;
const history = $('Refresh history').first().json;
const data = $('Data agent').first().json;
const modeling = $('Modeling agent').first().json;
const input = $('Read input').first().json;
const adjustments = [];

const METRIC = /^(rmse|rmsle|mae|roc_auc|log_loss|accuracy|f1|map@\d+)$/;
let metric = String(planner.metric ?? '').toLowerCase();
if (!METRIC.test(metric)) {
  adjustments.push(`metric '${planner.metric}' is not supported; using ${profile.suggested_metric}`);
  metric = profile.suggested_metric;
}

const columns = Object.keys(profile.features);
const knownColumns = (field) => {
  const requested = data[field] ?? [];
  const unknown = requested.filter((c) => !columns.includes(c));
  if (unknown.length) adjustments.push(`${field}: ignored unknown columns ${unknown.join(', ')}`);
  return requested.filter((c) => columns.includes(c));
};

const RANGES = {
  lightgbm: {
    learning_rate: [0.005, 0.3], num_leaves: [8, 512], max_depth: [-1, 16], min_child_samples: [5, 500],
    reg_alpha: [0, 100], reg_lambda: [0, 100], subsample: [0.3, 1], colsample_bytree: [0.2, 1],
    n_estimators: [100, 10000],
  },
  xgboost: {
    learning_rate: [0.005, 0.3], max_depth: [1, 16], min_child_weight: [0, 100], gamma: [0, 20],
    reg_alpha: [0, 100], reg_lambda: [0, 100], subsample: [0.3, 1], colsample_bytree: [0.2, 1],
    n_estimators: [100, 10000],
  },
};
const INTEGER_PARAMS = new Set(['num_leaves', 'max_depth', 'min_child_samples', 'n_estimators']);

const DEFAULTS = {
  lightgbm: { learning_rate: 0.05, num_leaves: 63, max_depth: -1, min_child_samples: 20, reg_alpha: 0,
    reg_lambda: 0, subsample: 0.8, colsample_bytree: 0.8, n_estimators: 3000 },
  xgboost: { learning_rate: 0.05, max_depth: 6, min_child_weight: 1, gamma: 0, reg_alpha: 0, reg_lambda: 1,
    subsample: 0.8, colsample_bytree: 0.8, n_estimators: 3000 },
};
// The training rows are part of the feature set: adding external data is a feature change.
const featureKey = (f, externalData) => JSON.stringify([
  [...(f.drop ?? [])].sort(), [...(f.frequency_encode ?? [])].sort(), [...(f.as_category ?? [])].sort(),
  [...(f.target_encode ?? [])].sort(), [...(f.digit_features ?? [])].sort(), f.add_null_count === true,
  [...(externalData ?? [])].sort(),
]);
const effective = (m, p) => ({ ...DEFAULTS[m], ...p });
const runs = [history.best_full_data_run, history.best_full_data_model_run, ...(history.recent ?? [])]
  .filter(Boolean).filter((r) => r.sample_frac === input.sample_frac);
const done = (config, duplicate) => [{
  json: { config, adjustments, duplicate_of: duplicate ? duplicate.run_id : null },
}];
const note = `iteration ${it.iteration}: ${it.plan.focus ?? ''}`;

// The first run of a competition is the plain baseline (default params, no feature
// changes, no external data): without it, whatever the agents add first cannot be
// told apart from the model's own score.
if (history.n_experiments === 0) {
  const asked = modeling.model === 'xgboost' ? 'xgboost' : 'lightgbm';
  adjustments.push(`no experiment yet in this competition: training the plain ${asked} baseline first, `
    + 'so the proposed changes have a reference to be measured against');
  return done({ metric, model: asked, params: {}, features: { drop: [], frequency_encode: [], as_category: [],
    target_encode: [], digit_features: [], add_null_count: false }, external_data: [],
  sample_frac: input.sample_frac, note: `${note} (baseline first)` }, null);
}

// A blend averages the predictions of earlier runs; only runs that saved their
// predictions and were trained on the same rows can take part.
const best = history.best_full_data_model_run;
let model = modeling.model;
if (model === 'blend') {
  const requested = [...new Set(modeling.blend_runs ?? [])];
  const usable = requested.filter((id) => runs.some((r) => r.run_id === id && r.blendable && r.model !== 'blend'));
  const rejected = requested.filter((id) => !usable.includes(id));
  if (rejected.length) {
    adjustments.push(`blend: ignored runs ${rejected.join(', ')} (unknown, a blend, another sample size or `
      + 'trained before predictions were saved)');
  }
  if (usable.length >= 2) {
    const key = JSON.stringify([...usable].sort());
    const tried = runs.find((r) => r.model === 'blend' && JSON.stringify([...(r.blend_runs ?? [])].sort()) === key);
    if (tried) adjustments.push(`blend repeats run ${tried.run_id} (cv ${tried.cv_score}); training skipped`);
    return done({ metric, model: 'blend', blend_runs: usable, sample_frac: input.sample_frac, note }, tried);
  }
  // A requested run that only lacks saved predictions is retrained as it was, so the
  // next iteration can blend it.
  const missing = runs.find((r) => rejected.includes(r.run_id) && !r.blendable && r.model !== 'blend');
  if (missing) {
    adjustments.push(`blend needs two usable runs; retraining run ${missing.run_id} as it was so it can be blended`);
    return done({ metric, model: missing.model, params: missing.params,
      features: { target_encode: [], digit_features: [], ...missing.features }, external_data: missing.external_data ?? [],
      sample_frac: input.sample_frac, note }, null);
  }
  adjustments.push('blend needs two usable runs; training the best single-model run\'s configuration instead');
  model = best?.model ?? 'lightgbm';
}

if (!RANGES[model]) {
  adjustments.push(`model '${model}' is not supported; using lightgbm`);
  model = 'lightgbm';
}

const params = {};
for (const [name, value] of Object.entries(modeling.params ?? {})) {
  const range = RANGES[model][name];
  if (!range || typeof value !== 'number' || !Number.isFinite(value)) {
    adjustments.push(`param '${name}' ignored (not allowed for ${model})`);
    continue;
  }
  let safe = Math.min(Math.max(value, range[0]), range[1]);
  if (INTEGER_PARAMS.has(name)) safe = Math.round(safe);
  if (safe !== value) adjustments.push(`param '${name}' adjusted ${value} -> ${safe}`);
  params[name] = safe;
}

// External datasets: only the ones discovery accepted for this competition.
const acceptedExternal = ($('Discover external data').first().json.accepted ?? []).map((d) => d.name);
const externalData = [...new Set(data.external_data ?? [])].filter((name) => {
  if (acceptedExternal.includes(name)) return true;
  adjustments.push(`external_data: ignored unknown dataset ${name}`);
  return false;
});

const hasNulls = Object.values(profile.features).some((f) => f.null_pct > 0);
if (data.add_null_count === true && !hasNulls) {
  adjustments.push('add_null_count disabled: the data has no missing values');
}

const features = {
  drop: knownColumns('drop'),
  frequency_encode: knownColumns('frequency_encode'),
  as_category: knownColumns('as_category'),
  target_encode: knownColumns('target_encode').filter((c) => !(data.drop ?? []).includes(c)),
  digit_features: knownColumns('digit_features').filter((c) => profile.features[c]?.min !== undefined),
  add_null_count: data.add_null_count === true && hasNulls,
};

// Attribution: one kind of change per iteration relative to a reference run, so
// the effect on the score can be read from the result. The reference is the best
// run of the proposed model with the proposed features if one exists (then up to
// two hyperparameters may change), else the best run of that model (or the best
// single-model run if the model was never trained). If the features differ from
// the reference, the params stay as in the reference.
const LOWER_IS_BETTER = /^(rmse|rmsle|mae|log_loss)$/;
const bestOf = (list) => list.reduce((a, b) => {
  if (!a) return b;
  const better = LOWER_IS_BETTER.test(metric) ? b.cv_score < a.cv_score : b.cv_score > a.cv_score;
  return better ? b : a;
}, null);
const ofModel = runs.filter((r) => r.model === model);
const ref = bestOf(ofModel.filter((r) => featureKey(r.features, r.external_data) === featureKey(features, externalData))) ?? bestOf(ofModel) ?? best;
if (ref && RANGES[ref.model]) {
  const refParams = effective(ref.model, ref.params);
  if (featureKey(features, externalData) !== featureKey(ref.features, ref.external_data)) {
    const ownParams = effective(model, params);
    const paramsChanged = model !== ref.model
      || Object.keys(refParams).some((k) => ownParams[k] !== refParams[k]);
    if (paramsChanged) {
      adjustments.push(`features changed vs reference run ${ref.run_id}: model and params reset to that run's `
        + `(${ref.model} ${JSON.stringify(ref.params)}) to isolate the feature effect`);
      model = ref.model;
      for (const k of Object.keys(params)) delete params[k];
      Object.assign(params, ref.params);
    }
  } else if (model === ref.model) {
    const ownParams = effective(model, params);
    const changed = Object.keys(ownParams).filter((k) => ownParams[k] !== refParams[k]);
    for (const k of changed.slice(2)) {
      adjustments.push(`param '${k}' reverted to the reference run's value ${refParams[k]}: at most two `
        + `hyperparameter changes per iteration`);
      if (k in ref.params) params[k] = ref.params[k];
      else delete params[k];
    }
  }
}

// Repeats add no information, except retraining a run from before predictions
// were saved, which makes it available for blending.
const matches = runs.filter((r) => r.model === model && featureKey(r.features, r.external_data) === featureKey(features, externalData)
  && JSON.stringify(effective(r.model, r.params)) === JSON.stringify(effective(model, params)));
const tried = matches.find((r) => r.blendable) ?? null;
if (tried) {
  adjustments.push(`configuration repeats run ${tried.run_id} (cv ${tried.cv_score}); training skipped`);
} else if (matches.length) {
  adjustments.push(`configuration repeats run ${matches[0].run_id}, which has no saved predictions; retrained so it can be blended`);
}

return done({ metric, model, params, features, external_data: externalData, sample_frac: input.sample_frac, note }, tried);
