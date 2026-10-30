const test = require('node:test');
const assert = require('node:assert/strict');
const { runNode } = require('./harness');

// What n8n's error branch actually carries for Groq 429 and 404 responses.
const limited = 'The service is receiving too many requests from you';
const missing = 'The resource you are requesting could not be found';
const env = { GROQ_FALLBACK_MODEL: 'small-model,other-model' };

// Runs the failures in order, each pass seeing the previous one as n8n does.
function failures(...errors) {
  let previous = null;
  return errors.map((error, runIndex) => {
    previous = runNode('note-failure.js', previous ? { 'Note failure': previous } : {}, { json: { error }, env, runIndex });
    return previous;
  });
}

test('a first rate limit is waited out on the same model', () => {
  const [out] = failures(limited);
  assert.equal(out.failures, 1);
  assert.equal(out.level, 0);
  assert.equal(out.use_fallback, false);
});

test('a rate limit that survives the wait is the daily quota: next model in the chain', () => {
  const [, out] = failures(limited, limited);
  assert.equal(out.level, 1);
  assert.match(out.fallback_reason, /^rate limit persisted/);
});

test('each model in the chain gets its own wait before the next one', () => {
  const outs = failures(limited, limited, limited, limited, limited);
  assert.deepEqual(outs.map((o) => o.level), [0, 1, 1, 2, 2]);
});

test('a missing model moves on at once', () => {
  const [out] = failures(missing);
  assert.equal(out.level, 1);
  assert.match(out.fallback_reason, /^model unavailable/);
});

test('an output the parser rejects is retried on the same model', () => {
  const [, out] = failures(limited, 'Failed to parse. Text: "{". Error: Unexpected end of JSON input');
  assert.equal(out.level, 0);
  assert.equal(out.streak, 0);
});
