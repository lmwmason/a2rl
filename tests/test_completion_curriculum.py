from types import SimpleNamespace

import torch

from tools import evaluate_rgb_imu_race as evaluator
from tools.evaluate_rgb_imu_race import completion_profile, succeeds
from training.curriculum import FINISH_MARKER, PACE_MARKER, SPEED_MARKER


def test_focus_uses_recent_failures_and_releases_reliably_cleared_gates():
    old = [{"gates": 12, "status": 1}] * 10
    recent = [{"gates": gates, "status": 4 if gates == 24 else 1}
              for gates in (0, 1, 2, 2, 2, 3, 3, 24)]
    assert completion_profile(old + recent) == {"focus_gate": 2, "reliable_gates": 2, "evaluation_count": 8}


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
    (run / "speed").mkdir()
    recovery = run / "recovery/latest.pt"
    torch.save({"training_progress": {"curriculum_stage": "finish"}}, recovery)
    torch.save({"training_progress": {"steps": 600}}, run / "speed/latest.pt")
    args = SimpleNamespace(local_run=run, out=out, public=None, adaptive_completion=False,
                           completion_rollouts=False, rollout_corrections=None, launch_corrections=None)
    checked = {"finish": [], "pace": [], "speed": []}

    def evaluate(_, weights, session, side, correction, **kwargs):
        stage = session.relative_to(out).parts[0]
        checkpoint = torch.load(weights, weights_only=True)["training_progress"].get("curriculum_stage")
        checked[stage].append((checkpoint, side))
        if stage == "pace":
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
