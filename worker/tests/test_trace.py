import json

import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture
def client(data_root):
    main._loaded.clear()
    return TestClient(main.app)


def test_attempt_records_agent_decisions_and_tool_calls(client, data_root):
    attempt = client.post("/attempts", json={"competition": "binary-comp", "goal": "top 20%"}).json()
    attempt_id = attempt["attempt_id"]
    headers = {"X-Attempt-Id": attempt_id}

    client.post(f"/attempts/{attempt_id}/events",
                json={"agent": "planner", "content": {"metric": "roc_auc"}, "reasoning": "binary target, proba sample"})
    client.get("/competitions/binary-comp/profile", headers=headers)
    run = client.post("/competitions/binary-comp/train", headers=headers,
                      json={"metric": "roc_auc", "params": {"n_estimators": 20}, "n_folds": 2}).json()
    client.post("/competitions/binary-comp/train", headers=headers, json={"metric": "rmse"})  # agent mistake

    finished = client.post(f"/attempts/{attempt_id}/finish", json={"summary": {"best_run": run["run_id"]}}).json()

    kinds = [(e["agent"], e["kind"]) for e in finished["events"]]
    assert kinds == [
        ("planner", "decision"),
        ("worker", "tool_call"), ("worker", "tool_result"),
        ("worker", "tool_call"), ("worker", "tool_result"),
        ("worker", "tool_call"), ("worker", "error"),
    ]
    assert finished["status"] == "completed" and finished["summary"]["best_run"] == run["run_id"]
    assert run["attempt_id"] == attempt_id

    exported = data_root / "runs" / "binary-comp" / "attempts" / f"{attempt_id}.jsonl"
    assert len(exported.read_text().splitlines()) == len(kinds)
    assert json.loads(exported.read_text().splitlines()[0])["reasoning"] == "binary target, proba sample"
    index = [json.loads(line) for line in exported.with_name("index.jsonl").read_text().splitlines()]
    assert [(a["attempt_id"], a["status"]) for a in index] == [(attempt_id, "completed")]

    assert client.get("/attempts", params={"competition": "binary-comp"}).json()[0]["attempt_id"] == attempt_id


def test_failed_execution_closes_its_attempt(client):
    attempt = client.post("/attempts", json={"competition": "binary-comp", "execution_id": "42"}).json()

    failed = client.post("/executions/42/fail", json={"node": "Planner", "error": "401 Invalid API Key"}).json()

    assert failed["attempt_id"] == attempt["attempt_id"] and failed["status"] == "failed"
    assert failed["summary"] == {"failed_node": "Planner", "error": "401 Invalid API Key"}
    assert failed["events"][-1]["kind"] == "error"


def test_unknown_attempt_is_rejected(client):
    response = client.post("/attempts/nope/events", json={"agent": "planner", "content": "x"})
    assert response.status_code == 422


def test_decision_keeps_what_the_agent_saw_and_what_the_call_cost(client, data_root):
    attempt_id = client.post("/attempts", json={"competition": "binary-comp"}).json()["attempt_id"]
    context = {"profile": {"task": "binary"}, "history": {"n_experiments": 0}}
    meta = {"model": "openai/gpt-oss-120b", "llm_attempts": 2, "elapsed_ms": 4100,
            "token_usage": {"promptTokens": 3673, "completionTokens": 658, "totalTokens": 4331}}
    client.post(f"/attempts/{attempt_id}/events", json={
        "agent": "planner", "content": {"metric": "roc_auc"}, "reasoning": "binary", "context": context, "meta": meta})

    event = client.get(f"/attempts/{attempt_id}").json()["events"][0]
    assert event["context"] == context and event["meta"] == meta

    client.post(f"/attempts/{attempt_id}/finish", json={})
    exported = data_root / "runs" / "binary-comp" / "attempts" / f"{attempt_id}.jsonl"
    assert json.loads(exported.read_text().splitlines()[0])["meta"]["token_usage"]["totalTokens"] == 4331


def test_silent_attempts_are_closed_as_orphaned(client):
    from datetime import datetime, timedelta, timezone

    from app import trace

    quiet = client.post("/attempts", json={"competition": "binary-comp"}).json()["attempt_id"]
    client.post(f"/attempts/{quiet}/events", json={"agent": "planner", "content": "x"})

    assert trace.sweep_stale() == []  # recent activity: still running
    assert trace.sweep_stale(datetime.now(timezone.utc) + timedelta(hours=1)) == [quiet]

    attempt = client.get(f"/attempts/{quiet}").json()
    assert attempt["status"] == "orphaned" and attempt["summary"]["last_event_at"]
    assert attempt["events"][-1]["kind"] == "error"


def test_unexpected_errors_keep_their_traceback(data_root, monkeypatch):
    def broken(*_, **__):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.training.train", broken)
    client = TestClient(main.app, raise_server_exceptions=False)
    attempt_id = client.post("/attempts", json={"competition": "binary-comp"}).json()["attempt_id"]

    response = client.post("/competitions/binary-comp/train", headers={"X-Attempt-Id": attempt_id},
                           json={"metric": "roc_auc"})

    assert response.status_code == 500 and "boom" in response.json()["error"]
    error = client.get(f"/attempts/{attempt_id}").json()["events"][-1]["content"]
    assert "RuntimeError: boom" in error["traceback"] and "broken" in error["traceback"]
