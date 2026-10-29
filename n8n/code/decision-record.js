// The trace event for this decision: what the agent decided, what it saw (its
// full context) and what the LLM call cost (model, attempts, time, prompt size).
// Token counts stay inside the LangChain sub-node, which workflow code cannot
// read, so the exact prompt size in characters is logged instead.
// __AGENT__, __DEFAULT_MODEL__ and __SYSTEM_CHARS__ are filled in by build_workflows.py.
const failures = $('Note failure').isExecuted ? $('Note failure').first().json.failures : 0;
const context = $('Input').first().json.context;

return [{
  json: {
    agent: '__AGENT__',
    kind: 'decision',
    content: $json.output,
    reasoning: $json.output.reasoning,
    context,
    meta: {
      model: $env.GROQ_MODEL || '__DEFAULT_MODEL__',
      llm_attempts: failures + 1,
      elapsed_ms: Date.now() - $('Start clock').first().json.started_ms,
      prompt_chars: __SYSTEM_CHARS__ + 'Context:\n'.length + JSON.stringify(context).length,
    },
  },
}];
