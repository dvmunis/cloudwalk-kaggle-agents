// Counts failed LLM calls in this decision ($runIndex grows on each pass) and
// decides whether the next call switches to the fallback model.
// n8n hands this branch only its own summary of the Groq error ("The service is
// receiving too many requests from you"), not Groq's text, so a spent daily quota
// cannot be told apart from the per-minute limit by its message. It is told apart
// by persistence instead: the per-minute window has passed after one wait, the
// daily one has not. A missing or withdrawn model switches at once. The switch
// sticks for the rest of this decision.
const text = (error) => String(error?.message ?? error ?? '');
const RATE_LIMIT = /too many requests|rate limit|tokens per day|\(TPD\)/i;
const error = $json.error;
const message = text(error);
const modelGone = /could not be found|model_not_found|decommissioned|does not exist/i.test(message);
const previous = $runIndex > 0 ? $('Note failure').last().json : null;
const limitPersisted = RATE_LIMIT.test(message) && previous !== null && RATE_LIMIT.test(text(previous.error));

let reason = previous?.use_fallback ? previous.fallback_reason : null;
if (!reason && modelGone) reason = `model unavailable: ${message.slice(0, 200)}`;
if (!reason && limitPersisted) reason = `rate limit persisted after waiting (daily quota): ${message.slice(0, 200)}`;

return [{ json: { failures: $runIndex + 1, error, use_fallback: reason !== null, fallback_reason: reason } }];
