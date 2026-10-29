// What the Planner sees: the goal, the data profile and what has been tried before.
const input = $('Read input').first().json;
return [{
  json: {
    attempt_id: $('Start attempt').first().json.attempt_id,
    context: {
      goal: input.goal,
      iterations_in_this_attempt: input.max_iterations,
      profile: $('Profile data').first().json,
      history: $('Get history').first().json,
    },
  },
}];
