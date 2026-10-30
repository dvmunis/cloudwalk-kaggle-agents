const test = require('node:test');
const assert = require('node:assert/strict');
const { runNode } = require('./harness');

// What n8n's error branch actually carries for Groq 429 and 404 responses.
const limited = 'The service is receiving too many requests from you';
const missing = 'The resource you are requesting could not be found';

const noteFailure = (error, previous) => runNode('note-failure.js', previous ? { 'Note failure': previous } : {},
  { json: { error }, runIndex: previous ? previous.failures : 0 });

test('a first rate limit is waited out on the same model', () => {
  const out = noteFailure(limited);
  assert.equal(out.failures, 1);
  assert.equal(out.use_fallback, false);
});

test('a rate limit that survives the wait is the daily quota: switch to the fallback', () => {
  const out = noteFailure(limited, { failures: 1, error: limited, use_fallback: false, fallback_reason: null });
  assert.equal(out.use_fallback, true);
  assert.match(out.fallback_reason, /^rate limit persisted/);
});

test('a missing model switches at once', () => {
  const out = noteFailure(missing);
  assert.equal(out.use_fallback, true);
  assert.match(out.fallback_reason, /^model unavailable/);
});

test('once switched, later failures keep the fallback and the original reason', () => {
  const out = noteFailure(limited, { failures: 2, error: limited, use_fallback: true, fallback_reason: 'model unavailable: x' });
  assert.equal(out.use_fallback, true);
  assert.equal(out.fallback_reason, 'model unavailable: x');
});

test('an output the parser rejects is retried on the same model', () => {
  const out = noteFailure('Failed to parse. Text: "{". Error: Unexpected end of JSON input',
    { failures: 1, error: limited, use_fallback: false, fallback_reason: null });
  assert.equal(out.use_fallback, false);
});
