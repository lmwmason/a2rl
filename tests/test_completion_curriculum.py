from types import SimpleNamespace
import json

import torch

from tools import evaluate_rgb_imu_race as evaluator
from tools.evaluate_rgb_imu_race import completion_profile, handoff_schedule, replay_timing, succeeds
from tools.replay_export import race_summary
from training.curriculum import FINISH_MARKER, PACE_MARKER, SPEED_MARKER, SPEED_RUN
from training.live_feed import status
from training.speed_elodin import perturb
from training.race_reward import GateTrace, matched_improvement, reward_terms


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


def test_replay_timing_uses_recorded_crossings_and_ends_at_finish_or_contact():
    race = {"finished": True, "gates": 24, "gates_per_race": 24,
            "gate_times": list(range(2, 26)), "video_start_seconds": .2, "end_seconds": 27.}
    assert replay_timing(race) == {"video_start_seconds": .2, "start_seconds": 2, "gate_seconds": list(range(2, 26)),
                                   "lap_end_seconds": [13, 25], "stop_seconds": 25, "laps": 2}
    crash = race | {"finished": False, "gates": 14, "gate_times": race["gate_times"][:14], "end_seconds": 16.2}
    assert replay_timing(crash)["lap_end_seconds"] == [13]
    assert replay_timing(crash)["stop_seconds"] == 16.2
    assert replay_timing(crash | {"gates": 0, "gate_times": []})["start_seconds"] is None
    assert replay_timing(race | {"gate_times": None}) is None


def test_incomplete_replay_keeps_only_completed_lap_splits():
    replay = {"status": 1, "gates": 23, "gates_per_race": 24, "official_seconds": None,
              "timing": {"start_seconds": 2., "lap_end_seconds": [16.],
                         "stop_seconds": 31., "video_start_seconds": 0.}}
    title, label, total, rows = race_summary(replay, 35.)
    assert (title, label, total) == ("Crash · 23 / 24 gates", "Elapsed time", "29.000 s")
    assert rows[1:] == (("1", "14.000 s", "14.000 s"), ("2", "—", "—"))


def test_speed_score_prefers_clean_progress_and_a_faster_finish():
    finish = {"gates": 24, "finished": True, "contact": False, "gate_times": list(range(24)),
              "official_seconds": 24., "reward_trace": {"segment_progress": 0., "slow_approach_seconds": 0.,
                                                       "crossing_offsets_m": [0.] * 24}}
    crash = finish | {"gates": 23, "finished": False, "contact": True, "gate_times": list(range(23))}
    def score(result):
        return sum(reward_terms(result, finish).values())

    assert score(finish | {"official_seconds": 23.}) - score(finish) == 1.75
    assert reward_terms(crash, finish)["terminal"] == -18.
    assert score(finish) > score(crash) > score(crash | {"gates": 2, "gate_times": [1, 2]})
    slow = finish | {"reward_trace": finish["reward_trace"] | {"slow_approach_seconds": 1.}}
    assert score(slow) < score(finish)
    wide = finish | {"reward_trace": finish["reward_trace"] | {"crossing_offsets_m": [.53] * 24}}
    assert score(wide) < score(finish)
    assert matched_improvement([finish | {"official_seconds": 23.}] * 3, [finish] * 3)
    assert matched_improvement([finish | {"official_seconds": 24.07}, finish, finish], [finish, crash, crash])
    assert not matched_improvement([finish | {"official_seconds": 24.2}, finish, finish], [finish] * 3)
    assert not matched_improvement([finish, crash, finish], [finish] * 3)


def test_reward_trace_records_stopping_near_gate_and_crossing_offset():
    gate = SimpleNamespace(center=(0., 0., 0.), yaw_deg=0.)
    trace = GateTrace((-4., 0., 0.), [gate])
    trace.observe(0, (-1., 0., 0.), (-1., 0., 0.), 1.)
    trace.observe(0, (-1., 0., 0.), (-1., 0., 0.), 1.1)
    assert trace.report()["segment_progress"] == .75
    assert trace.report()["slow_approach_seconds"] > .09
    trace.observe(0, (-.01, .4, 0.), (.01, .4, 0.), 1.2, crossed=True)
    assert trace.report()["crossing_offsets_m"] == [.4]
    assert trace.report()["segment_progress"] == 0.


def test_gate_exit_progress_scores_only_completed_transitions():
    gates = [SimpleNamespace(center=(0., 0., z), yaw_deg=0.) for z in (0., 2.)]
    trace = GateTrace((-4., 0., 0.), gates)
    trace.observe(0, (-.01, 0., 0.), (.01, 0., 0.), 1., crossed=True)
    trace.observe(1, (.1, 0., 0.), (0., 0., 1.), 1.36)
    trace.observe(1, (-.01, 0., 2.), (.01, 0., 2.), 2., crossed=True)
    report = trace.report()
    assert report["exit_progress_m"] == [1., None]

    result = {"gates": 2, "finished": False, "contact": False, "gate_times": [1., 2.],
              "reward_trace": report}
    reference = result | {"reward_trace": report | {"exit_progress_m": [0., None]}}
    assert reward_terms(result, reference, exit_progress_weight=.35)["gate_exit"] > 0.
    assert "gate_exit" not in reward_terms(result, reference)
    missed = result | {"gates": 1, "contact": True, "gate_times": [1.]}
    assert reward_terms(missed, reference, exit_progress_weight=.35)["gate_exit"] == 0.


def test_speed_policy_perturbations_are_paired_and_keep_the_encoders():
    tensor = torch.ones((2, 2))
    policy = {"encoder": {"weight": tensor},
              "actor": {"0.weight": tensor, "2.weight": tensor},
              "future_actor": {"weight": tensor}}
    variants = [perturb(policy, torch.Generator().manual_seed(9), .01, sign) for sign in (-1, 1)]
    for trial in variants:
        assert trial["encoder"] is policy["encoder"]
        assert trial["actor"]["0.weight"] is tensor
        assert not torch.equal(trial["actor"]["2.weight"], tensor)
    for name, key in (("actor", "2.weight"), ("future_actor", "weight")):
        torch.testing.assert_close((variants[0][name][key] + variants[1][name][key]) / 2, tensor)


def test_speed_status_does_not_reuse_imitation_statistics(tmp_path, monkeypatch):
    (tmp_path / PACE_MARKER).touch()
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    (recovery / "metrics.jsonl").write_text('{"round": 99, "steps": 99}')
    monkeypatch.setattr("training.live_feed.training_active", lambda _: True)
    live = status(tmp_path, tmp_path)
    assert live["starting_speed_trainer"] and live["active"]
    assert live["phase"] == "SPEED RL" and live["steps"] is None and live["rollouts"] == 0
    speed = tmp_path / SPEED_RUN
    speed.mkdir()
    (speed / "latest.pt").touch()
    (speed / "metrics.jsonl").write_text('{"round": 1, "steps": 500, "rollouts": 3, "mean_return": 4.5}')
    live = status(tmp_path, tmp_path)
    assert not live["starting_speed_trainer"]
    assert (live["round"], live["steps"], live["rollouts"], live["mean_return"]) == (1, 500, 3, 4.5)


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
