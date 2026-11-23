from tools.evaluate_rgb_imu_race import completion_profile, succeeds


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
