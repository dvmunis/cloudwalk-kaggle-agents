// What the Evaluator sees: the run that just finished and the best earlier full-data run.
const run = $('Train').first().json;
return [{
  json: {
    attempt_id: $('Start attempt').first().json.attempt_id,
    context: {
      run: {
        run_id: run.run_id,
        metric: run.metric,
        higher_is_better: run.higher_is_better,
        cv_score: run.cv_score,
        cv_std: run.cv_std,
        fold_scores: run.fold_scores,
        best_iterations: run.best_iterations,
        sample_frac: run.config.sample_frac,
        model: run.config.model,
        params: run.config.params,
        features: run.config.features,
        top_features: run.top_features.slice(0, 8),
        blend: run.blend,
      },
      best_previous_full_data_run: $('Refresh history').first().json.best_full_data_run,
    },
  },
}];
