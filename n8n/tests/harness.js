// Runs an Orchestrator Code node outside n8n. Node outputs are given by name;
// $(name).first() and .last() return them and $(name).isExecuted is true for given names,
// as in n8n for the node's latest run.
const fs = require('node:fs');
const path = require('node:path');

function runNode(file, outputs, { json = {}, env = {}, runIndex = 0 } = {}) {
  const code = fs.readFileSync(path.join(__dirname, '..', 'code', file), 'utf8');
  const item = (name) => (name in outputs ? { json: outputs[name] } : undefined);
  const $ = (name) => ({ isExecuted: name in outputs, first: () => item(name), last: () => item(name) });
  const result = new Function('$', '$json', '$env', '$runIndex', code)($, json, env, runIndex);
  return result[0].json;
}

module.exports = { runNode };
