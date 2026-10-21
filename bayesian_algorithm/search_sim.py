"""
search_sim.py

Offline Monte-Carlo test of the search logic. No ROS, no Gazebo: a hovering drone, a target moving in a
straight line (optionally turning), a gimbal that follows the same model the planner assumes, and a detector
that fires with the SAME P_D the planner uses. It therefore tests the LOGIC (does the loop find the target,
and faster than a raster?), not the realism of P_D - Gazebo does that.

Every strategy sees identical trials (same seeds, same starting situation).

    python3 search_sim.py                  # 30 trials per strategy
    python3 search_sim.py --trials 100 --turn
"""

import argparse
import math
from statistics import median

import numpy as np

from camera_model import Camera
from search_core import Config, SearchCore

DRONE_POS = (778.0, 788.0, 75.0)        # hovering drone, roughly mid-map
DRONE_Q = (0.0, 0.0, 0.0, 1.0)
TRACK_S, SIM_DT, FRAME_DT = 20.0, 0.05, 0.1
SUCCESS_RADIUS = 40.0                   # metres: a found position this close to the truth counts as correct


def run_trial(strategy, seed, cfg, cam, turn=False, gap=(10.0, 30.0), vel_bias=3.0):
    rng = np.random.default_rng(seed)                       # trial randomness (same for every strategy)
    core = SearchCore(cfg, cam, strategy, seed=seed + 1)
    core.set_drone_pose(DRONE_POS, DRONE_Q)

    # target starts 80-250 m from the drone, random heading, 5 m/s +-15%
    r, th = rng.uniform(80, 250), rng.uniform(0, 2 * math.pi)
    pos = np.array([DRONE_POS[0] + r * math.cos(th), DRONE_POS[1] + r * math.sin(th)])
    hd, spd = rng.uniform(0, 2 * math.pi), 5.0 * rng.uniform(0.85, 1.15)
    vel = spd * np.array([math.cos(hd), math.sin(hd)])
    t_turn = rng.uniform(25, 60) if turn else None
    turn_sign = rng.choice([-1.0, 1.0])
    det_rng = np.random.default_rng(seed + 2)

    # phase 1: normal tracking, noisy measurements at 10 Hz
    t = 0.0
    while t < TRACK_S:
        t += FRAME_DT
        pos = pos + vel * FRAME_DT
        core.on_track_measurement(t, pos[0] + rng.normal(0, cfg.meas_noise_std), pos[1] + rng.normal(0, cfg.meas_noise_std))
    last_seen_pos = pos.copy()
    pitch0, yaw0 = cam.aim_angles(last_seen_pos, DRONE_POS, DRONE_Q, cfg.pitch_min, cfg.pitch_max)
    core.kf.x[2:4] += rng.normal(0, vel_bias, 2)           # real logs showed a few m/s of velocity error at the loss moment

    # phase 2: the target is unseen for a while (occlusion / forced loss), then the search starts
    gap_s = rng.uniform(*gap)
    while t < TRACK_S + cfg.t_lost + gap_s:
        t += SIM_DT
        pos = pos + vel * SIM_DT
    assert core.declare_loss(t, pitch0, yaw0)
    next_frame = t

    # phase 3: search
    while core.state == "SEARCH":
        t += SIM_DT
        if t_turn is not None and t - TRACK_S >= t_turn:    # one 90-degree turn at a random time
            c, s = math.cos(turn_sign * math.pi / 2), math.sin(turn_sign * math.pi / 2)
            vel = np.array([c * vel[0] - s * vel[1], s * vel[0] + c * vel[1]])
            t_turn = None
        pos = pos + vel * SIM_DT
        core.tick(t)
        if t >= next_frame and core.state == "SEARCH":
            next_frame += FRAME_DT
            pd = cam.detection_probability(pos[None, :], DRONE_POS, DRONE_Q, core.gimbal.pitch, core.gimbal.yaw)[0]
            dets = []
            if det_rng.random() < pd:
                px, py, _, _ = cam.ground_to_pixel(pos[None, :], DRONE_POS, DRONE_Q, core.gimbal.pitch, core.gimbal.yaw)
                dets.append({"cx": float(px[0]), "cy": float(py[0]), "conf": 0.8})
            core.on_detections(t, dets)
    ep = core.last_episode
    ok = ep["outcome"] == "REACQUIRED" and math.hypot(ep["found_xy"][0] - pos[0], ep["found_xy"][1] - pos[1]) < SUCCESS_RADIUS
    false = ep["outcome"] == "REACQUIRED" and not ok
    return {"ok": ok, "false": false, "time": ep["duration"], "deg": ep["gimbal_deg"],
            "ms": ep["compute_ms"], "views": ep["views"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--turn", action="store_true", help="target makes one 90 degree turn after loss")
    ap.add_argument("--max-search", type=float, default=90.0)
    ap.add_argument("--gap", type=float, nargs=2, default=[10.0, 30.0], metavar=("LO", "HI"),
                    help="seconds the target is unseen before the search starts")
    ap.add_argument("--vel-bias", type=float, default=3.0, help="std of the velocity error at loss, m/s per axis")
    ap.add_argument("--strategies", default="bayes,raster,spiral,predict")
    ap.add_argument("--ablations", action="store_true", help="also run the three ablations of the proposed method")
    a = ap.parse_args()

    cam = Camera()
    runs = [(s, {}) for s in a.strategies.split(",")]
    if a.ablations:
        runs += [("bayes", {"ablate_prediction": True}), ("bayes", {"ablate_miss": True}), ("bayes", {"lam": 0.0})]

    print(f"{a.trials} trials per row, timeout {a.max_search:.0f} s, turn={'yes' if a.turn else 'no'}")
    print(f"{'method':<24}{'success':>9}{'false':>7}{'timeout':>9}{'median t (s)':>14}{'mean t (s)':>12}{'gimbal deg':>12}{'plan ms':>9}")
    for strat, over in runs:
        cfg = Config(max_search=a.max_search, **over)
        res = [run_trial(strat, 1000 + i, cfg, cam, a.turn, tuple(a.gap), a.vel_bias) for i in range(a.trials)]
        n = len(res)
        times = [r["time"] if r["ok"] else a.max_search for r in res]          # timeouts and false matches count as the limit
        labels = {"ablate_prediction": "no motion prediction", "ablate_miss": "no miss update", "lam": "no movement penalty"}
        name = strat + ("".join(f" [{labels[k]}]" for k in over) if over else "")
        print(f"{name:<24}{100 * sum(r['ok'] for r in res) / n:8.0f}%{100 * sum(r['false'] for r in res) / n:6.0f}%"
              f"{100 * sum((not r['ok']) and (not r['false']) for r in res) / n:8.0f}%{median(times):14.1f}{np.mean(times):12.1f}"
              f"{np.mean([r['deg'] for r in res]):12.0f}{np.mean([r['ms'] for r in res]):9.0f}")


if __name__ == "__main__":
    main()
    