from types import SimpleNamespace
import json

import torch

from tools import evaluate_rgb_imu_race as evaluator
from tools.evaluate_rgb_imu_race import completion_profile, handoff_schedule, succeeds
from training.curriculum import FINISH_MARKER, PACE_MARKER, SPEED_MARKER, SPEED_RUN


def test_focus_uses_recent_failures_and_releases_reliably_cleared_gates():
    old = [{"gates": 12, "status": 1}] * 10
    recent = [{"gates": gates, "status": 4 if gates == 24 else 1}
              for gates in (0, 1, 2, 2, 2, 3, 3, 24)]
    assert completion_profile(old + recent) == {"focus_gate": 2, "reliable_gates": 2, "evaluation_count": 8}


def test_gate_handoff_moves_earlier_after_two_clear_segments(tmp_path):
    log = tmp_path / "handoff.jsonl"
    independent = [{"gates": 4, "status": 1}] * 5
    assert handoff_schedule(independent, log) == {"after_gates": 3, "target_gates": 5}
    result = {"after_gates": 3, "target_gates": 5, "gates": 5, "contact": False, "student_frames": 30}
    log.write_text((json.dumps(result) + "\n") * 2)
    assert handoff_schedule(independent, log) == {"after_gates": 2, "target_gates": 5}
    log.write_text((json.dumps(result | {"gates": 4, "contact": True}) + "\n") * 3)
    assert handoff_schedule(independent, log) == {"after_gates": 4, "target_gates": 5}
    log.write_text((json.dumps(result | {"after_gates": 4, "gates": 4, "contact": True}) + "\n") * 3)
    assert handoff_schedule(independent, log) == {"after_gates": 4, "target_gates": 5, "delay_s": .15}


def test_preserve_best_measured_checkpoint_when_latest_regresses(tmp_path):
    four = {"gates": 4, "contact": True, "finished": False, "official_seconds": None}
    evaluator.retain_independent_candidate(tmp_path, b"four", four, {"round": 30})
    evaluator.retain_independent_candidate(tmp_path, b"three", four | {"gates": 3}, {"round": 31})
    assert (tmp_path / "best-independent.pt").read_bytes() == b"four"
    assert json.loads((tmp_path / "best-independent.json").read_text())["round"] == 30


def test_assisted_finish_never_qualifies():
    finish = {"physics_engine": "elodin", "finished": True, "contact": False, "gates": 24,
              "gates_per_race": 24, "official_seconds": 29.1}
    assert succeeds(finish, 30.)
    assert not succeeds(finish | {"assisted": True}, 30.)


def test_curriculum_requires_a_pace_trained_checkpoint_before_rl(tmp_path, monkeypatch):
    run = tmp_path / "run"
    out = tmp_path / "out"
    out.mkdir()
    (run / "recovery").mkdir(parents=True)
    (run / SPEED_RUN).mkdir()
    recovery = run / "recovery/latest.pt"
    torch.save({"training_progress": {"curriculum_stage": "finish"}}, recovery)
    torch.save({"training_progress": {"steps": 600, "physics_engine": "elodin"}}, run / SPEED_RUN / "latest.pt")
    (run / "speed").mkdir()
    torch.save({"training_progress": {"steps": 99}}, run / "speed/latest.pt")
    args = SimpleNamespace(local_run=run, out=out, public=None, adaptive_completion=False,
                           completion_rollouts=True, rollout_corrections=tmp_path, launch_corrections=None)
    checked = {"finish": [], "pace": [], "speed": []}

    def evaluate(_, weights, session, side, correction, **kwargs):
        stage = session.relative_to(out).parts[0]
        if kwargs.get("assisted"):
            assert stage == "finish"
            return {"assisted": True}
        checkpoint = torch.load(weights, weights_only=True)["training_progress"].get("curriculum_stage")
        checked[stage].append((checkpoint, side))
        if stage == "pace":
            assert kwargs["seconds"] > 30
            assert kwargs["max_gate_gap"] == 10.
            suffix = "" if side == 0 else "-start-1" if side < 0 else "-start-2"
            assert correction.name.endswith(f"-pace{suffix}.npz")
            assert (run / FINISH_MARKER).is_file()
            assert not (run / PACE_MARKER).is_file()
            if checkpoint == "finish":
                torch.save({"training_progress": {"curriculum_stage": "pace"}}, recovery)
        if stage == "speed":
            assert (run / PACE_MARKER).is_file()
        return {"physics_engine": "elodin", "finished": True, "contact": False, "gates": 24,
                "gates_per_race": 24, "official_seconds": {"finish": 42, "pace": 29, "speed": 11}[stage]}

    monkeypatch.setattr(evaluator, "evaluate", evaluate)
    evaluator.watch(args)
    assert checked["finish"] == [("finish", side) for side in evaluator.STARTS]
    assert checked["pace"] == [(stage, side) for stage in ("finish", "pace") for side in evaluator.STARTS]
    assert checked["speed"] == [(None, side) for side in evaluator.STARTS]
    assert (run / SPEED_MARKER).is_file()
