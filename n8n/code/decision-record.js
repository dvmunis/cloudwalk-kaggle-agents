// The trace event for this decision: what the agent decided, what it saw (its
// full context) and what the LLM call cost (model, attempts, time, prompt size).
// Token counts stay inside the LangChain sub-node, which workflow code cannot
// read, so the exact prompt size in characters is logged instead.
// __AGENT__, __MODEL__ and __SYSTEM_CHARS__ are filled in by build_workflows.py;
// __MODEL__ is the same expression that picks the chat model (primary or fallback).
const failed = $('Note failure').isExecuted ? $('Note failure').last().json : null;
const context = $('Input').first().json.context;

return [{
  json: {
    agent: '__AGENT__',
    kind: 'decision',
    content: $json.output,
    reasoning: $json.output.reasoning,
    context,
    meta: {
      model: __MODEL__,
      fallback_reason: failed?.use_fallback ? failed.fallback_reason : null,
      llm_attempts: (failed?.failures ?? 0) + 1,
      elapsed_ms: Date.now() - $('Start clock').first().json.started_ms,
      prompt_chars: __SYSTEM_CHARS__ + 'Context:\n'.length + JSON.stringify(context).length,
    },
  },
}];
