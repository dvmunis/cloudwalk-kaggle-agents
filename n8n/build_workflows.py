"""Generate the n8n workflows: one sub-workflow per agent, plus the Orchestrator.

Every agent has the same shape, so the role-specific parts (system prompt and
JSON Schema of the decision) live here and the workflow JSON is generated:

    Input (called by the Orchestrator) -> LLM chain (Groq + structured output)
      -> log decision to the worker trace -> return the decision

If the LLM call fails (usually Groq's free-tier tokens-per-minute limit), the
agent waits and calls it again, a few times, before failing the attempt.

The Orchestrator wires the agents to the worker's tools. The JavaScript of its
Code nodes lives in n8n/code/ so it can be read and reviewed as code.

Run after editing a prompt or a code file:  python3 n8n/build_workflows.py
"""

import json
import uuid
from pathlib import Path

OUT_DIR = Path(__file__).parent / "workflows"
CODE_DIR = Path(__file__).parent / "code"
GROQ_CREDENTIAL = {"id": "kgGroqCredent01", "name": "Groq"}
DEFAULT_MODEL = "openai/gpt-oss-120b"
# Used in order once the model before cannot serve this call any more (daily quota
# spent, model withdrawn); each has its own quota on Groq's free tier.
FALLBACK_CHAIN = "openai/gpt-oss-20b,qwen/qwen3.8-27b"
LLM_RETRIES = 6
LLM_RETRY_WAIT_S = 30

SHARED_RULES = (
    "You are part of an autonomous team of agents that competes in Kaggle tabular competitions. "
    "Other agents act on your decision without human review, so be concrete, use only the options "
    "listed, and never invent column names, metrics or parameters. Explain your decision in "
    "`reasoning` in 2-4 sentences that cite the evidence from the context.\n\n"
    "What the training tool already does, so never ask for it again: text (object) columns are passed "
    "to the models as native categorical features; training is 5-fold cross-validation with early "
    "stopping. Defaults: lightgbm learning_rate 0.05, num_leaves 63, subsample 0.8, colsample_bytree "
    "0.8; xgboost learning_rate 0.05, max_depth 6, subsample 0.8, colsample_bytree 0.8; n_estimators "
    "3000 as an upper bound for both. Frequency encoding only helps columns with many distinct values.\n\n"
    "Team tools: feature operations (drop columns, frequency encoding, target encoding, numeric columns as "
    "categorical, digit features, per-row null count), external data, model choice (lightgbm or xgboost) with hyperparameters, "
    "and blends. External data: public datasets with the same columns as the competition (for synthetic "
    "Playground competitions, the original dataset they were generated from); their rows are added to the "
    "training folds only, so the CV score stays comparable, and adding them often helps more than tuning, "
    "so when one is available try it early. Numeric columns with few distinct values (n_unique in the "
    "profile, roughly under 100) are usually levels or codes rather than measurements: as categorical "
    "they often help tree models, most of all in multiclass problems. "
    "Target encoding adds the smoothed mean of the target per value of a column, computed out-of-fold so it "
    "does not leak; it helps categorical or discrete columns whose values carry signal. A blend averages the "
    "predictions of two or more earlier runs (history entries with blendable true) without retraining and is "
    "scored by cross-validation like any run; it gains most when the runs are strong and different, e.g. a "
    "lightgbm and an xgboost run, or runs with different features. A per-row null count only helps when the "
    "data has missing values; never plan it otherwise."
)

AGENTS = {
    "planner": {
        "id": "kgAgentPlanner01",
        "title": "Planner",
        "system": (
            "Role: Planner. You receive the data profile of a competition, the team goal and the history "
            "of past experiments. Decide the evaluation metric and a short strategy for this attempt.\n"
            "- metric must be one of: rmse, rmsle, mae, roc_auc, log_loss, accuracy, f1, map@K (e.g. map@3). "
            "The profile's suggested_metric was inferred from the submission file; keep it unless the "
            "evidence clearly points elsewhere.\n"
            "- strategy: at most 4 steps, each achievable with the team's tools.\n"
            "- With history available, build on the best run instead of starting over.\n"
            "- external_data_queries: 2-3 short Kaggle dataset searches that could find the real-world dataset "
            "behind this competition, written from the domain the columns describe (e.g. 'customer churn "
            "telecom'), not from the competition name. The team downloads candidates and keeps only those "
            "with the same columns.\n"
            "- The attempt runs several iterations; each iteration is one training run that changes one "
            "thing (features, or up to two hyperparameters, or the model, or a blend of earlier runs). "
            "Order the strategy so the "
            "iterations can follow it, and set focus_for_this_attempt to the first iteration's change."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "metric": {"type": "string"},
                "strategy": {"type": "array", "items": {"type": "string"}},
                "focus_for_this_attempt": {"type": "string"},
                "external_data_queries": {"type": "array", "items": {"type": "string"}},
                "reasoning": {"type": "string"},
            },
            "required": ["metric", "strategy", "focus_for_this_attempt", "external_data_queries", "reasoning"],
        },
    },
    "data": {
        "id": "kgAgentData0001",
        "title": "Data",
        "system": (
            "Role: Data agent. Given the data profile, the current plan and focus, and the experiment "
            "history, choose the feature operations for this iteration's training run.\n"
            "- drop: columns to remove (e.g. pure identifiers or leakage).\n"
            "- frequency_encode: columns that get an extra <col>_freq feature with value counts.\n"
            "- as_category: numeric columns with few distinct values that behave like categories.\n"
            "- target_encode: columns that get an extra <col>_te feature with the out-of-fold target mean; it "
            "works for numeric columns with many distinct values too.\n"
            "- digit_features: numeric columns that get features from how the value is written (last digits, "
            "first decimal, multiple of 100); synthetic generators leave traces there.\n"
            "- external_data: names from external_datasets to add as training rows (empty for none).\n"
            "- add_null_count: true to add a per-row count of missing values (only useful with nulls).\n"
            "Use only column names that appear in profile.features. Start from the features of "
            "history.best_full_data_model_run (if any) and change at most one thing. If plan.focus is about the "
            "model, hyperparameters or a blend, return the best run's features unchanged, so the effect of the "
            "change can be attributed. Never propose a feature set already tried in the history. Empty "
            "lists and add_null_count false are a valid answer."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "drop": {"type": "array", "items": {"type": "string"}},
                "frequency_encode": {"type": "array", "items": {"type": "string"}},
                "as_category": {"type": "array", "items": {"type": "string"}},
                "target_encode": {"type": "array", "items": {"type": "string"}},
                "digit_features": {"type": "array", "items": {"type": "string"}},
                "add_null_count": {"type": "boolean"},
                "external_data": {"type": "array", "items": {"type": "string"}},
                "reasoning": {"type": "string"},
            },
            "required": ["drop", "frequency_encode", "as_category", "target_encode", "digit_features", "add_null_count",
                         "external_data", "reasoning"],
        },
    },
    "modeling": {
        "id": "kgAgentModeling1",
        "title": "Modeling",
        "system": (
            "Role: Modeling agent. Choose the model and hyperparameters for the next training run, or a blend.\n"
            "- model: lightgbm, xgboost or blend.\n"
            "- blend: set blend_runs to 2-5 run_ids from history.recent with blendable true and model not "
            "blend, picking strong runs that differ from each other; params and features are ignored. Use it "
            "when plan.focus asks for a blend. Never repeat a blend already in the history.\n"
            "- lightgbm params: learning_rate, num_leaves, max_depth, min_child_samples, reg_alpha, "
            "reg_lambda, subsample, colsample_bytree, n_estimators.\n"
            "- xgboost params: learning_rate, max_depth, min_child_weight, gamma, reg_alpha, reg_lambda, "
            "subsample, colsample_bytree, n_estimators.\n"
            "Training uses 5-fold CV with early stopping, so n_estimators is only an upper bound. "
            "Otherwise start from the best run in the history of the model you choose (or "
            "history.best_full_data_model_run). If feature_config differs from that run's features, keep that "
            "run's params unchanged. Otherwise "
            "change at most two hyperparameters, or switch the model, following plan.focus; never repeat "
            "a configuration already in the history. A guardrail enforces these rules, so breaking them "
            "only wastes the iteration. Omitted params keep the defaults."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "model": {"type": "string", "enum": ["lightgbm", "xgboost", "blend"]},
                "params": {"type": "object", "additionalProperties": {"type": "number"}},
                "blend_runs": {"type": "array", "items": {"type": "string"}},
                "reasoning": {"type": "string"},
            },
            "required": ["model", "params", "blend_runs", "reasoning"],
        },
    },
    "evaluator": {
        "id": "kgAgentEvaluat01",
        "title": "Evaluator",
        "system": (
            "Role: Evaluator. You receive the result of the training run that just finished and the best "
            "previous full-data run (null if none). Decide whether this run is worth a Kaggle submission.\n"
            "- Submissions are limited per day: submit only if this is the first full-data run, or if it "
            "beats the best previous full-data run by more than that run's cv_std. Respect "
            "higher_is_better.\n"
            "- Never submit a run trained on a sample (sample_frac < 1).\n"
            "- verdict 'suspicious' if fold scores vary widely or the score is implausibly good.\n"
            "- next_focus: one concrete idea for the next experiment."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "submit": {"type": "boolean"},
                "verdict": {"type": "string", "enum": ["first_baseline", "improved", "no_improvement", "suspicious"]},
                "next_focus": {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["submit", "verdict", "next_focus", "reasoning"],
        },
    },
    "reviewer": {
        "id": "kgAgentLBReview1",
        "title": "LB Reviewer",
        "system": (
            "Role: Leaderboard Reviewer. An iteration just finished. You receive its run (and its public "
            "score if the Evaluator submitted it), the team's position on the public leaderboard (from its "
            "best submission so far), the gap to the top-20% threshold, the iterations of this attempt and "
            "the plan the team is following. Review the plan and propose the next one.\n"
            "- If the run was submitted, compare public_score with cv_score: a public score far below the CV "
            "(more than about 3x cv_std) means the validation is not trustworthy; say so and favour more "
            "regularized models.\n"
            "- Judge the plan by the evidence: which changes moved the score and by how much compared with "
            "the gap to the goal. Drop directions that did not help.\n"
            "- plan: at most 4 steps, each one iteration that changes one thing with the team's tools. "
            "Put the step with the largest expected gain first. When tuning one model gives gains far "
            "smaller than the gap, prefer bigger levers: target encoding, a different model family, and "
            "blends of strong, diverse runs.\n"
            "- focus_for_next_iteration: the single concrete change for the next iteration.\n"
            "- stop: true only if the goal is reached or no change available to the team could plausibly "
            "close the gap."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "assessment": {"type": "string"},
                "plan": {"type": "array", "items": {"type": "string"}},
                "focus_for_next_iteration": {"type": "string"},
                "stop": {"type": "boolean"},
                "reasoning": {"type": "string"},
            },
            "required": ["assessment", "plan", "focus_for_next_iteration", "stop", "reasoning"],
        },
    },
}


def _node_id(workflow_id: str, node_name: str) -> str:
    # Deterministic, so regenerating a workflow does not churn its diff.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{workflow_id}/{node_name}"))


def _connect(*names: str) -> dict:
    """Main connections for a linear chain of nodes."""
    return {a: {"main": [[{"node": b, "type": "main", "index": 0}]]} for a, b in zip(names, names[1:])}


# Which model the next LLM call uses: the primary, or the model of the fallback chain
# that "Note failure" moved to (shared by the chat model and the trace).
MODEL_EXPR = (f"(($('Note failure').isExecuted ? $('Note failure').last().json.level : 0) > 0)"
              f" ? ($env.GROQ_FALLBACK_MODEL || '{FALLBACK_CHAIN}').split(',')"
              f"[$('Note failure').last().json.level - 1].trim()"
              f" : ($env.GROQ_MODEL || '{DEFAULT_MODEL}')")


def build_agent(name: str, spec: dict) -> dict:
    title = spec["title"]
    nid = lambda node_name: _node_id(spec["id"], node_name)  # noqa: E731
    record_js = ((CODE_DIR / "decision-record.js").read_text()
                 .replace("__AGENT__", name).replace("__DEFAULT_MODEL__", DEFAULT_MODEL)
                 .replace("__SYSTEM_CHARS__", str(len(f"{SHARED_RULES}\n\n{spec['system']}")))
                .replace("__MODEL__", MODEL_EXPR))
    nodes = [
        {
            "name": "Input", "type": "n8n-nodes-base.executeWorkflowTrigger", "typeVersion": 1.1,
            "position": [0, 0], "parameters": {"inputSource": "passthrough"},
        },
        {
            "name": "Start clock", "type": "n8n-nodes-base.code", "typeVersion": 2, "position": [200, 0],
            "parameters": {"jsCode": "return [{ json: { started_ms: Date.now() } }];"},
        },
        {
            "name": f"{title} Agent", "type": "@n8n/n8n-nodes-langchain.chainLlm", "typeVersion": 1.7,
            "position": [420, 0],
            "retryOnFail": True, "maxTries": 2, "waitBetweenTries": 5000, "onError": "continueErrorOutput",
            "parameters": {
                "promptType": "define",
                "text": "={{ 'Context:\\n' + JSON.stringify($('Input').first().json.context) }}",
                "hasOutputParser": True,
                "messages": {"messageValues": [
                    {"type": "SystemMessagePromptTemplate", "message": f"{SHARED_RULES}\n\n{spec['system']}"}
                ]},
            },
        },
        {
            "name": "Groq Chat Model", "type": "@n8n/n8n-nodes-langchain.lmChatGroq", "typeVersion": 1,
            "position": [360, 220],
            "parameters": {
                "model": f"={{{{ {MODEL_EXPR} }}}}",
                "options": {"temperature": 0.2},
            },
            "credentials": {"groqApi": GROQ_CREDENTIAL},
        },
        {
            "name": "Decision Schema", "type": "@n8n/n8n-nodes-langchain.outputParserStructured",
            "typeVersion": 1.3, "position": [560, 220],
            "parameters": {"schemaType": "manual", "inputSchema": json.dumps(spec["schema"], indent=2)},
        },
        _code_text("Decision record", record_js, [700, 0]),
        # A worker blip must not lose the decision: the log call is retried.
        {**_http("Log decision", "POST", "={{ $env.WORKER_URL }}/attempts/{{ $('Input').first().json.attempt_id }}/events",
                 body="$json"),
         "position": [940, 0], "retryOnFail": True, "maxTries": 3, "waitBetweenTries": 3000},
        {
            "name": "Return decision", "type": "n8n-nodes-base.code", "typeVersion": 2, "position": [1180, 0],
            "parameters": {"jsCode": f"return [{{ json: $('{title} Agent').first().json.output }}];"},
        },
        _code_text("Note failure", (CODE_DIR / "note-failure.js").read_text()
                   .replace("__FALLBACK_CHAIN__", FALLBACK_CHAIN), [700, 400]),
        {
            "name": "Retry?", "type": "n8n-nodes-base.if", "typeVersion": 2, "position": [940, 400],
            "parameters": {
                "conditions": {
                    "options": {"caseSensitive": True, "leftValue": "", "typeValidation": "strict"},
                    "conditions": [{
                        "id": nid("Retry?/condition"),
                        "leftValue": f"={{{{ $json.failures <= {LLM_RETRIES} }}}}",
                        "rightValue": "",
                        "operator": {"type": "boolean", "operation": "true", "singleValue": True},
                    }],
                    "combinator": "and",
                },
                "options": {},
            },
        },
        {
            "name": "Wait", "type": "n8n-nodes-base.wait", "typeVersion": 1.1, "position": [1180, 400],
            "webhookId": nid("Wait/webhook"), "parameters": {"amount": LLM_RETRY_WAIT_S},
        },
        {
            "name": "Give up", "type": "n8n-nodes-base.stopAndError", "typeVersion": 1, "position": [1180, 600],
            "parameters": {"errorMessage": f"={{{{ '{title} agent: LLM call failed after {LLM_RETRIES} "
                                           "retries: ' + ($json.error?.message ?? $json.error ?? 'unknown error') }}"},
        },
    ]
    for node in nodes:
        node["id"] = nid(node["name"])
    return {
        "id": spec["id"],
        "name": f"Kaggle Team — {title} Agent",
        "active": False,
        "settings": {"executionOrder": "v1", "callerPolicy": "workflowsFromSameOwner"},
        "nodes": nodes,
        "connections": {
            **_connect("Input", "Start clock", f"{title} Agent"),
            f"{title} Agent": {"main": [
                [{"node": "Decision record", "type": "main", "index": 0}],
                [{"node": "Note failure", "type": "main", "index": 0}],
            ]},
            **_connect("Decision record", "Log decision", "Return decision"),
            **_connect("Note failure", "Retry?"),
            "Retry?": {"main": [
                [{"node": "Wait", "type": "main", "index": 0}],
                [{"node": "Give up", "type": "main", "index": 0}],
            ]},
            **_connect("Wait", f"{title} Agent"),
            "Groq Chat Model": {"ai_languageModel": [[{"node": f"{title} Agent", "type": "ai_languageModel", "index": 0}]]},
            "Decision Schema": {"ai_outputParser": [[{"node": f"{title} Agent", "type": "ai_outputParser", "index": 0}]]},
        },
        "pinData": {},
    }


ORCHESTRATOR_ID = "kgOrchestrator01"
ERROR_HANDLER_ID = "kgErrorHandler01"
ATTEMPT_HEADER = {
    "sendHeaders": True,
    "headerParameters": {
        "parameters": [{"name": "X-Attempt-Id", "value": "={{ $('Start attempt').first().json.attempt_id }}"}]
    },
}
COMPETITION_URL = "={{ $env.WORKER_URL }}/competitions/{{ $('Read input').first().json.competition }}"
ATTEMPT_URL = "={{ $env.WORKER_URL }}/attempts/{{ $('Start attempt').first().json.attempt_id }}"


def _http(name, method, url, body=None, traced=False, timeout_ms=None):
    params = {"method": method, "url": url}
    if traced:
        params.update(ATTEMPT_HEADER)
    if body is not None:
        params.update({"sendBody": True, "specifyBody": "json", "jsonBody": f"={{{{ JSON.stringify({body}) }}}}"})
    params["options"] = {"timeout": timeout_ms} if timeout_ms else {}
    return {"name": name, "type": "n8n-nodes-base.httpRequest", "typeVersion": 4.2, "parameters": params}


def _code_text(name, js, position=None):
    node = {"name": name, "type": "n8n-nodes-base.code", "typeVersion": 2, "parameters": {"jsCode": js}}
    if position:
        node["position"] = position
    return node


def _code(name, file):
    return {"name": name, "type": "n8n-nodes-base.code", "typeVersion": 2,
            "parameters": {"jsCode": (CODE_DIR / file).read_text()}}


def _call_agent(name, agent):
    return {
        "name": name, "type": "n8n-nodes-base.executeWorkflow", "typeVersion": 1.2,
        "parameters": {
            "source": "database",
            "workflowId": {"__rl": True, "value": AGENTS[agent]["id"], "mode": "id"},
            "workflowInputs": {"mappingMode": "defineBelow", "value": {}, "matchingColumns": [], "schema": [],
                               "attemptToConvertTypes": False, "convertFieldsToString": True},
            "options": {"waitForSubWorkflow": True},
        },
    }


def _if(name, condition):
    return {"name": name, "type": "n8n-nodes-base.if", "typeVersion": 2, "parameters": {
        "conditions": {
            "options": {"caseSensitive": True, "leftValue": "", "typeValidation": "strict"},
            "conditions": [{
                "id": _node_id(ORCHESTRATOR_ID, f"{name}/condition"),
                "leftValue": condition,
                "rightValue": "",
                "operator": {"type": "boolean", "operation": "true", "singleValue": True},
            }],
            "combinator": "and",
        },
        "options": {},
    }}


def _link(*targets):
    """One output per argument; each output goes to a node name or a list of names."""
    return {"main": [[{"node": t, "type": "main", "index": 0} for t in ([o] if isinstance(o, str) else o)]
                     for o in targets]}


def build_orchestrator() -> dict:
    """Setup once, then iterations of Data -> Modeling -> train -> Evaluator [-> submit] -> LB Reviewer.

    The loop is a connection from "Next iteration?" back to "Iteration context".
    Nodes inside the loop read other nodes with .first(), which returns their
    latest run, i.e. the current iteration.
    """
    setup = [
        {"name": "Start", "type": "n8n-nodes-base.webhook", "typeVersion": 2,
         "webhookId": "7d1c6a52-2f4b-4c8e-9a51-6b0f3e9d1a01",
         "parameters": {"httpMethod": "POST", "path": "kaggle-team", "responseMode": "responseNode", "options": {}}},
        _code("Read input", "read-input.js"),
        _http("Start attempt", "POST", "={{ $env.WORKER_URL }}/attempts",
              body="{ competition: $json.competition, goal: $json.goal, execution_id: $execution.id }"),
        _http("Download data", "POST", COMPETITION_URL + "/download", traced=True, timeout_ms=600000),
        _http("Profile data", "GET", COMPETITION_URL + "/profile", traced=True, timeout_ms=300000),
        _http("Get history", "GET", COMPETITION_URL + "/history"),
        _code("Planner context", "planner-context.js"),
        _call_agent("Planner", "planner"),
        # A failed search must not end the attempt: the agents then simply see no external data.
        {**_http("Discover external data", "POST", COMPETITION_URL + "/external/discover", traced=True,
                 timeout_ms=900000, body="{ queries: $('Planner').first().json.external_data_queries ?? [] }"),
         "onError": "continueRegularOutput"},
    ]
    iteration = [
        _code("Iteration context", "iteration-context.js"),
        _http("Refresh history", "GET", COMPETITION_URL + "/history?limit=40"),
        _code("Data context", "data-context.js"),
        _call_agent("Data agent", "data"),
        _code("Modeling context", "modeling-context.js"),
        _call_agent("Modeling agent", "modeling"),
        _code("Build train config", "build-train-config.js"),
        _http("Log guardrails", "POST", ATTEMPT_URL + "/events",
              body="{ agent: 'guardrails', kind: 'message', content: $json, reasoning: $json.adjustments.length "
                   "? 'Agent output corrected to the allowed options' : 'Agent output accepted as is' }"),
        _if("New config?", "={{ !$('Build train config').first().json.duplicate_of }}"),
        _http("Train", "POST", COMPETITION_URL + "/train", traced=True, timeout_ms=1800000,
              body="$('Build train config').first().json.config"),
        _code("Evaluator context", "evaluator-context.js"),
        _call_agent("Evaluator", "evaluator"),
        _if("Submit?", "={{ $('Read input').first().json.submit && $json.submit === true }}"),
    ]
    review = [
        {**_http("Submit", "POST", COMPETITION_URL + "/submit", traced=True, timeout_ms=600000,
                 body="{ run_id: $('Train').first().json.run_id, message: $('Evaluator').first().json.verdict }"),
         "onError": "continueErrorOutput"},
        {**_http("Leaderboard", "GET", COMPETITION_URL + "/leaderboard", traced=True, timeout_ms=300000),
         "onError": "continueErrorOutput"},
        _code("Reviewer context", "reviewer-context.js"),
        _call_agent("LB Reviewer", "reviewer"),
    ]
    loop_end = [
        _code("Record iteration", "record-iteration.js"),
        _http("Log iteration", "POST", ATTEMPT_URL + "/events",
              body="{ agent: 'orchestrator', kind: 'message', content: $json.record, reasoning: $json.continue "
                   "? 'Next iteration' : 'Stopping: ' + $json.stop_reason }"),
        _if("Next iteration?", "={{ $('Record iteration').first().json.continue === true }}"),
    ]
    closing = [
        _code("Summarize", "summarize.js"),
        _http("Finish attempt", "POST", ATTEMPT_URL + "/finish", body="{ status: 'completed', summary: $json.summary }"),
        {"name": "Respond", "type": "n8n-nodes-base.respondToWebhook", "typeVersion": 1.1, "parameters": {
            "respondWith": "json",
            "responseBody": "={{ JSON.stringify({ attempt_id: $json.attempt_id, status: $json.status, "
                            "summary: $json.summary, events: $json.events.length }) }}",
            "options": {},
        }},
    ]

    # Layout: setup on the top row, the loop below it, the review branch above
    # the end of the loop and the closing steps after it.
    for i, node in enumerate(setup):
        node["position"] = [220 * i, 0]
    for i, node in enumerate(iteration):
        node["position"] = [220 * i, 320]
    x0 = 220 * len(iteration)
    for i, node in enumerate(review):
        node["position"] = [x0 + 220 * i, 160]
    for i, node in enumerate(loop_end):
        node["position"] = [x0 + 220 * (len(review) + i), 320]
    for i, node in enumerate(closing):
        node["position"] = [x0 + 220 * (len(review) + len(loop_end) + i), 520]
    nodes = setup + iteration + review + loop_end + closing
    for node in nodes:
        node["id"] = _node_id(ORCHESTRATOR_ID, node["name"])
        # Trace writes are retried, so a worker blip does not lose them or end the attempt.
        if node["name"] in ("Log guardrails", "Log iteration", "Finish attempt"):
            node.update({"retryOnFail": True, "maxTries": 3, "waitBetweenTries": 3000})

    names = lambda path: [n["name"] for n in path]  # noqa: E731
    connections = {
        **_connect(*names(setup), "Iteration context"),
        **_connect(*names(iteration)),
        **_connect("Reviewer context", "LB Reviewer", "Record iteration"),
        **_connect(*names(loop_end)),
        **_connect(*names(closing)),
        "New config?": _link("Train", "Record iteration"),
        "Submit?": _link("Submit", "Leaderboard"),
        "Submit": _link("Leaderboard", "Leaderboard"),
        "Leaderboard": _link("Reviewer context", "Record iteration"),
        "Next iteration?": _link("Iteration context", "Summarize"),
    }
    return {
        "id": ORCHESTRATOR_ID,
        "name": "Kaggle Team — Orchestrator",
        "active": False,
        "settings": {"executionOrder": "v1", "errorWorkflow": ERROR_HANDLER_ID},
        "nodes": nodes,
        "connections": connections,
        "pinData": {},
    }


def build_error_handler() -> dict:
    """Runs when the Orchestrator fails: closes the attempt as failed, recording the node and error."""
    nodes = [
        {"name": "On orchestrator error", "type": "n8n-nodes-base.errorTrigger", "typeVersion": 1,
         "position": [0, 0], "parameters": {}},
        {**_http("Mark attempt failed", "POST",
                 "={{ $env.WORKER_URL }}/executions/{{ $json.execution.id }}/fail",
                 body="{ node: $json.execution.lastNodeExecuted, error: $json.execution.error.message }"),
         "position": [260, 0]},
    ]
    for node in nodes:
        node["id"] = _node_id(ERROR_HANDLER_ID, node["name"])
    return {
        "id": ERROR_HANDLER_ID,
        "name": "Kaggle Team — Error Handler",
        "active": False,
        "settings": {"executionOrder": "v1"},
        "nodes": nodes,
        "connections": _connect(*[n["name"] for n in nodes]),
        "pinData": {},
    }


def _write(path: Path, workflow: dict) -> None:
    path.write_text(json.dumps(workflow, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    for name, spec in AGENTS.items():
        _write(OUT_DIR / f"agent-{name}.json", build_agent(name, spec))
    _write(OUT_DIR / "orchestrator.json", build_orchestrator())
    _write(OUT_DIR / "error-handler.json", build_error_handler())
