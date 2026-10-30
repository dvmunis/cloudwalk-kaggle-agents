// Counts failed LLM calls in this decision ($runIndex grows on each pass) and
// decides which model the next call uses: level 0 is GROQ_MODEL, level n the n-th
// model in GROQ_FALLBACK_MODEL (a comma-separated chain).
// n8n hands this branch only its own summary of the Groq error ("The service is
// receiving too many requests from you"), not Groq's text, so a spent daily quota
// cannot be told apart from the per-minute limit by its message. It is told apart
// by persistence instead: the per-minute window has passed after one wait, the
// daily one has not. So a model moves to the next one in the chain after two rate
// limits in a row, or at once when the model is missing. The switch sticks for the
// rest of this decision.
const text = (error) => String(error?.message ?? error ?? '');
const RATE_LIMIT = /too many requests|rate limit|tokens per day|\(TPD\)/i;
const error = $json.error;
const message = text(error);
const modelGone = /could not be found|model_not_found|decommissioned|does not exist/i.test(message);
const previous = $runIndex > 0 ? $('Note failure').last().json : null;
const level = previous?.level ?? 0;
const streak = RATE_LIMIT.test(message) ? (previous?.streak ?? 0) + 1 : 0;
const chainLength = 1 + String($env.GROQ_FALLBACK_MODEL || '__FALLBACK_CHAIN__').split(',').length;

let next = level;
let reason = previous?.fallback_reason ?? null;
if ((modelGone || streak >= 2) && level + 1 < chainLength) {
  next = level + 1;
  reason = modelGone ? `model unavailable: ${message.slice(0, 200)}`
    : `rate limit persisted after waiting (daily quota): ${message.slice(0, 200)}`;
}

return [{
  json: {
    failures: $runIndex + 1, error, level: next, streak: next === level ? streak : 0,
    use_fallback: next > 0, fallback_reason: next > 0 ? reason : null,
  },
}];
