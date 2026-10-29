// What the Modeling agent sees: task, plan, chosen features and past runs.
const profile = $('Profile data').first().json;
const it = $('Iteration context').first().json;
// Runs trained on another share of the data are not comparable, so they are left out.
const history = $('Refresh history').first().json;
const sampleFrac = $('Read input').first().json.sample_frac;
return [{
  json: {
    attempt_id: it.attempt_id,
    context: {
      task: profile.task,
      train_rows: profile.train_rows,
      n_features: profile.n_features,
      iteration: `${it.iteration} of ${it.max_iterations}`,
      plan: it.plan,
      feature_config: $('Data agent').first().json,
      history: { ...history, recent: history.recent.filter((r) => r.sample_frac === sampleFrac) },
    },
  },
}];
