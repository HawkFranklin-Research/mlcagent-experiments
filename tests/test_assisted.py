"""Logic of assisted/run_assisted.py with a stub session: no model, no MLCA project."""

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_assisted", ROOT / "assisted/run_assisted.py")
assisted = importlib.util.module_from_spec(spec)
sys.modules["run_assisted"] = assisted
spec.loader.exec_module(assisted)

PLAN = {
    "cap_per_run": 3,
    "instructions": [
        {"id": "I1", "at_stage": "features", "text": "cluster  the\ntraining rows"},
        {"id": "I2", "at_stage": "model", "text": "fit the grid"},
    ],
    "repair": {"id": "R1", "text": "fix the failed step"},
}


class StubSession:
    def __init__(self, autopilot_reasons, stage="features"):
        self.state = SimpleNamespace(current_event=stage)
        self.principal = "user:policy/x"
        self.events, self.answers, self.customs = [], [], []
        self.reasons = list(autopilot_reasons)
        self.pending = []
        self.store = SimpleNamespace(read=list)
        self.decision_window = None

    def emit(self, kind, payload, **kw):
        self.events.append(kind)

    def questions(self):
        return self.pending

    def answer_question(self, qid, answer):
        self.answers.append((qid, answer))
        self.pending = []

    def custom(self, text):
        self.customs.append(text)
        return {"decision": "in_scope"}

    def autopilot(self, _):
        reason = self.reasons.pop(0)
        if reason == "failure":
            self.pending = [{"question_id": "q1", "recommended": 1, "options": [
                {"effect": {"kind": "pause"}}, {"effect": {"kind": "retry"}}]}]
        return {"reason": reason, "turn": []}


@contextmanager
def no_principal(session, human):
    yield


@pytest.fixture(autouse=True)
def plain_principal(monkeypatch):
    from mlca.autonomous import windows

    monkeypatch.setattr(windows, "principal", no_principal)


def test_instruction_given_once_per_stage_then_agent_choice():
    record = []
    window = assisted.make_window(PLAN, record)
    session = StubSession([])
    first = window.decide(session, "choice", 2, [{}, {}, {}])
    assert first == ({"custom": "cluster the training rows"}, True)
    assert window.decide(session, "choice", 2, [{}, {}, {}]) == (2, False)  # agent's own choice
    session.state.current_event = "model"
    assert window.decide(session, "choice", 0, [{}])[0] == {"custom": "fit the grid"}
    assert [r["id"] for r in record] == ["I1", "I2"] and record[0]["words"] == 4
    assert session.events.count("DecisionWindowHalted") == 2


def test_failure_pauses_and_gives_repair_then_continues():
    record = []
    window = assisted.make_window(PLAN, record)
    session = StubSession(["failure", "completed"], stage="validate")
    session.decision_window = window
    assisted.assisted_drive(session)
    assert session.answers == [("q1", 0)]  # "Pause", not the automatic retry
    assert session.customs == ["fix the failed step"]
    assert [r["id"] for r in record] == ["R1"] and record[0]["trigger"] == "failure"


def test_cap_ends_the_run_like_the_autonomous_runner():
    record = [{"id": f"X{i}", "words": 1} for i in range(3)]  # cap already used
    window = assisted.make_window(PLAN, record)
    session = StubSession(["failure"], stage="validate")
    session.decision_window = window
    from mlca.errors import MlcaUserError

    with pytest.raises(MlcaUserError, match="Autonomous run stopped: failure"):
        assisted.assisted_drive(session)
    assert session.customs == []
    assert window.decide(session, "choice", 1, [{}, {}]) == (1, False)


def test_instruction_files_are_valid():
    import yaml

    for name in ["instructions-hnscc.yaml", "instructions-smoke-gbsg2.yaml"]:
        plan = yaml.safe_load((ROOT / "assisted" / name).read_text())
        stages = [i["at_stage"] for i in plan["instructions"]]
        assert len(stages) == len(set(stages)) and len(stages) < plan["cap_per_run"]
        assert plan["repair"]["text"].strip()
