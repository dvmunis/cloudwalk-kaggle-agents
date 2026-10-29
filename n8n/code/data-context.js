// What the Data agent sees: the data profile, the current plan and focus, and past runs.
const it = $('Iteration context').first().json;
// Runs trained on another share of the data are not comparable, so they are left out.
const history = $('Refresh history').first().json;
const sampleFrac = $('Read input').first().json.sample_frac;
return [{
  json: {
    attempt_id: it.attempt_id,
    context: {
      iteration: `${it.iteration} of ${it.max_iterations}`,
      plan: it.plan,
      profile: $('Profile data').first().json,
      external_datasets: $('Discover external data').first().json.accepted ?? [],
      history: { ...history, recent: history.recent.filter((r) => r.sample_frac === sampleFrac) },
    },
  },
}];
