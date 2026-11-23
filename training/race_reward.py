"""Training-only race features and reward for the Elodin speed worker."""

import math


REWARD_VERSION = 2
EXIT_WINDOW_SECONDS = .35


def clean_finish(result):
    return result["finished"] and not result["contact"] and result["gates"] == 24


class GateTrace:
    def __init__(self, start, gates):
        self.centers = [item.center for item in gates]
        self.normals = [(math.cos(math.radians(item.yaw_deg)), math.sin(math.radians(item.yaw_deg)))
                        for item in gates]
        self.distances = [max(.01, math.dist(a, b))
                          for a, b in zip([start, *self.centers], self.centers)]
        self.previous = None
        self.slow_approach_seconds = 0.
        self.crossing_offsets_m = []
        self.exit_progress_m = []
        self.exit_sample = None
        self.segment_progress = 0.

    def observe(self, gate, before, after, seconds, *, crossed=False):
        if self.exit_sample is not None:
            index, start, crossing_time = self.exit_sample
            if seconds - crossing_time >= EXIT_WINDOW_SECONDS or crossed:
                # Measure three-dimensional motion, including stacked-gate transitions.
                target = self.centers[index + 1]
                self.exit_progress_m[index] = math.dist(start, target) - math.dist(after, target)
                self.exit_sample = None
        last = self.previous
        if not crossed and last is not None and last[0] == gate and seconds - last[1] < .049:
            return
        x, y, z = self.centers[gate]
        nx, ny = self.normals[gate]
        approach = -((after[0] - x) * nx + (after[1] - y) * ny)
        distance = math.dist(after, (x, y, z))
        self.segment_progress = max(0., min(.99, 1. - distance / self.distances[gate]))
        if last and last[0] == gate and .35 < approach < 2.5 and distance < 3.:
            dt = seconds - last[1]
            if 0. < dt <= .12:
                closing_speed = (last[2] - distance) / dt
                self.slow_approach_seconds += dt * max(0., min(1., 1. - closing_speed / 1.5))
        self.previous = (gate, seconds, distance)
        if crossed:
            normal_before = (before[0] - x) * nx + (before[1] - y) * ny
            fraction = max(0., min(1., -normal_before / max(1e-9, -approach - normal_before)))
            cx, cy, cz = (a + fraction * (b - a) for a, b in zip(before, after))
            self.crossing_offsets_m.append(max(abs(-(cx - x) * ny + (cy - y) * nx), abs(cz - z)))
            self.exit_progress_m.append(None)
            if gate + 1 < len(self.centers):
                self.exit_sample = gate, (cx, cy, cz), seconds
            self.segment_progress = 0.

    def report(self):
        return {"version": REWARD_VERSION, "segment_progress": self.segment_progress,
                "slow_approach_seconds": self.slow_approach_seconds,
                "exit_progress_m": self.exit_progress_m,
                "crossing_offsets_m": self.crossing_offsets_m}


def reward_terms(result, reference, *, exit_progress_weight=0., exit_progress_gates=()):
    trace = result["reward_trace"]
    actual = result["gate_times"]
    baseline = reference["gate_times"]
    actual_splits = [b - a for a, b in zip([0., *actual], actual)]
    reference_splits = [b - a for a, b in zip([0., *baseline], baseline)]
    split_gain = sum(.35 * math.tanh((ref - current) / max(.3, .30 * ref))
                     for current, ref in zip(actual_splits, reference_splits))
    edge_cost = sum(.25 * min(1., max(0., (offset - .35) / .18)) ** 2
                    for offset in trace["crossing_offsets_m"])
    finish = clean_finish(result)
    if finish:
        terminal = 32. - 1.75 * (result["official_seconds"] - 30.)
    elif result["contact"]:
        terminal = {"arena_boundary": -22., "invalid_state": -30.}.get(result.get("contact_kind"), -18.)
    else:
        terminal = -8. if result.get("stalled") else -6.
    terms = {
        "progress": 2. * result["gates"] + (0. if finish else 1.5 * trace["segment_progress"]),
        "split_pace": split_gain,
        "approach_flow": -.8 * trace["slow_approach_seconds"],
        "edge_clearance": -edge_cost,
        "terminal": terminal,
    }
    if exit_progress_weight:
        # A quick exit that subsequently misses the next gate earns no bonus.
        paired = max(0, min(result["gates"], reference["gates"]) - 1)
        advances = zip(trace["exit_progress_m"][:paired],
                       reference["reward_trace"]["exit_progress_m"][:paired])
        terms["gate_exit"] = exit_progress_weight * sum(
            math.tanh((current - previous) / .5)
            for index, (current, previous) in enumerate(advances)
            if (not exit_progress_gates or index % 12 + 1 in exit_progress_gates)
            and current is not None and previous is not None
        )
    return terms


def matched_improvement(candidate, incumbent):
    """Keep the incumbent unless speed improves across the matched clean starts."""
    if not candidate or len(candidate) != len(incumbent) or not all(map(clean_finish, candidate)):
        return False
    deltas = [base["official_seconds"] - trial["official_seconds"]
              for trial, base in zip(candidate, incumbent) if clean_finish(base)]
    if deltas and min(deltas) < -.15:
        return False
    return len(deltas) < len(candidate) or sum(deltas) / len(deltas) >= .015
