"""
search_core.py

The ROS-free search logic: loss handling, belief, planner, baselines, confirmation, bookkeeping.

States:   TRACK  -> (loss declared) -> SEARCH -> (confirmed or timed out) -> TRACK
SEARCH cycles through three phases:
    PLAN   choose where to point        (this is where the strategies differ)
    SLEW   gimbal moves and settles
    DWELL  watch the frames; if nothing is seen, apply ONE missed-detection update; then PLAN again

Strategies share everything except choose_aim(), so comparisons are like for like:
    bayes    maximise  sum_i b(i) P_D(i|a) - lambda * slew_time        (the proposed method)
    raster   serpentine sweep of ground aim points around the last known position
    spiral   expanding spiral of ground aim points around the last known position
    predict  keep pointing at the constant-velocity predicted position

Ground truth never enters this file. The caller scores the result.
"""

import math
import time
from dataclasses import dataclass

import numpy as np

from camera_model import Camera, clamp
from search_belief import SearchBelief
from target_kalman_filter import TargetKalmanFilter
from belief_grid import X_MIN, X_MAX, Y_MIN, Y_MAX, CELL_SIZE, POSITION_STD_FLOOR


@dataclass
class Config:
    # map and grid (real bounds live in belief_grid.py)
    bounds: tuple = (X_MIN, X_MAX, Y_MIN, Y_MAX)
    cell_size: float = CELL_SIZE
    # gimbal (slew rate and settle time are placeholders until a step response is measured)
    pitch_min: float = math.radians(-135.0)
    pitch_max: float = math.radians(45.0)
    slew_rate: float = math.radians(60.0)       # rad/s
    settle_s: float = 0.5                       # seconds to settle after arriving
    # loss and search limits
    t_lost: float = 2.0                         # seconds without a detection before declaring loss (matches gimbal_tracker)
    max_search: float = 90.0                    # timeout, seconds
    dwell_min_s: float = 0.4
    dwell_max_s: float = 2.0
    dwell_min_frames: int = 3
    frame_latency: float = 0.3                  # pipeline delay: frames older than settle + this are ignored
    # planner
    lam: float = 0.02                           # movement-time penalty (probability per second)
    n_top: int = 150                            # candidate aim cells: the most probable cells...
    n_stride: int = 150                         # ...plus a spread sample of the rest
    support_frac: float = 1e-3                  # cells below this fraction of the peak are ignored
    # baselines
    scan_spacing: float = 60.0                  # metres between aim points
    baseline_radius: float = 300.0              # search box half-width / spiral radius
    # confirmation
    conf_min: float = 0.5
    confirm_frames: int = 3
    confirm_radius: float = 30.0                # candidate positions must agree within this (m)
    gate_chi2: float = 16.0                     # candidate must lie within ~4 sigma of the motion prior
    # filter and belief
    process_noise_std: float = 0.5
    meas_noise_std: float = 8.0
    min_updates: int = 20                       # measurements needed before a loss can be declared
    pos_floor: float = POSITION_STD_FLOOR
    n_particles: int = 4000
    # ablations (proposed method only)
    ablate_prediction: bool = False             # belief frozen at the loss moment
    ablate_miss: bool = False                   # no missed-detection update


class GimbalModel:
    """
    Predicts the gimbal's actual angle. The sim publishes no angle feedback, so we command a rate-limited
    ramp and assume the real gimbal follows it. Angles are kept in [-pi, pi]: the joint is driven to the
    absolute value, so a wrapped command can swing the long way round, and slew time reflects that.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset(0.0, 0.0, 0.0)

    def reset(self, pitch, yaw, t):
        self.pitch, self.yaw, self.tp, self.ty = pitch, yaw, pitch, yaw
        self.t, self.ready_t, self.moved = t, t, 0.0

    def update(self, t):
        dt = max(0.0, t - self.t)
        step = self.cfg.slew_rate * dt
        for name, tgt in (("pitch", self.tp), ("yaw", self.ty)):
            cur = getattr(self, name)
            new = cur + clamp(tgt - cur, -step, step)
            self.moved += abs(new - cur)
            setattr(self, name, new)
        self.t = t

    def time_to_reach(self, pitch, yaw):
        dist = max(abs(clamp(pitch, self.cfg.pitch_min, self.cfg.pitch_max) - self.pitch), abs(yaw - self.yaw))
        return dist / self.cfg.slew_rate + self.cfg.settle_s

    def command(self, pitch, yaw, t):
        self.update(t)
        self.ready_t = t + self.time_to_reach(pitch, yaw)       # when it will have arrived AND settled
        self.tp = clamp(pitch, self.cfg.pitch_min, self.cfg.pitch_max)
        self.ty = yaw

    def ready(self, t):
        return t >= self.ready_t


class SearchCore:
    def __init__(self, cfg=None, cam=None, strategy="bayes", seed=None):
        self.cfg = cfg or Config()
        self.cam = cam or Camera()
        self.strategy = strategy
        self.rng = np.random.default_rng(seed)
        self.belief = SearchBelief(self.cfg.n_particles, self.cfg.process_noise_std, self.cfg.pos_floor,
                                   self.cfg.bounds, self.cfg.cell_size, seed)
        self.gimbal = GimbalModel(self.cfg)
        self.drone_pos = None
        self.drone_q = (0.0, 0.0, 0.0, 1.0)
        self.episodes = []                      # one dict per finished search
        self.log_events = []                    # events waiting to be collected by tick()
        self._new_filter()
        self.state = "TRACK"

    # ------------------------------------------------------------------ TRACK
    def _new_filter(self):
        self.kf = TargetKalmanFilter(self.cfg.process_noise_std, self.cfg.meas_noise_std)
        self.n_meas = 0

    def set_drone_pose(self, pos, q):
        self.drone_pos, self.drone_q = tuple(pos), tuple(q)

    def on_track_measurement(self, t, x, y):
        """A ground-position measurement from the estimator while the target is being tracked."""
        if self.state == "TRACK":
            self.kf.update(x, y, t)
            self.n_meas += 1

    def ready_for_loss(self):
        return self.state == "TRACK" and self.n_meas >= self.cfg.min_updates

    def declare_loss(self, t, gimbal_pitch, gimbal_yaw):
        """
        FREEZE POINT. Copies the filter's last valid state into the belief and starts the search.
        The caller decides WHEN (see the loss monitor in reacq_search_node.py, or a forced loss).
        """
        if not self.ready_for_loss() or self.drone_pos is None:
            return False
        self.snap_x, self.snap_P = self.kf.get_state()
        self.snap_t = self.kf.last_timestamp
        self.belief.start(self.snap_x, self.snap_P, self.snap_t)
        self.gimbal.reset(gimbal_pitch, gimbal_yaw, t)
        self.state, self.phase = "SEARCH", "PLAN"
        self.t_loss = t
        self.views = 0
        self.compute_ms = []
        self.log_events = []
        self.aim = None
        self.streak, self.last_g = 0, None
        centre, _, _ = self.prior(t)
        self.queue = self._scan_points(centre) if self.strategy in ("raster", "spiral") else []
        self.qi = 0
        self.log_events.append(("LOSS", t, f"snapshot age {t - self.snap_t:.1f}s at ({centre[0]:.0f},{centre[1]:.0f})"))
        return True

    def prior(self, t):
        """Closed-form motion prior from the snapshot: mean, position covariance (with floor), velocity."""
        kf = TargetKalmanFilter(self.cfg.process_noise_std)
        kf.x, kf.P = self.snap_x.copy(), self.snap_P.copy()
        kf.predict(max(0.0, t - self.snap_t))
        return kf.x[:2].copy(), kf.P[:2, :2] + self.cfg.pos_floor ** 2 * np.eye(2), kf.x[2:].copy()

    # ------------------------------------------------------------------ SEARCH
    def tick(self, t):
        """Advance the search. Returns {'cmd': (pitch, yaw) or None, 'events': [...]}."""
        out = {"cmd": None, "events": []}
        if self.state != "SEARCH" or self.drone_pos is None:
            out["events"], self.log_events = self.log_events, []     # e.g. a REACQUIRED raised inside on_detections()
            return out
        self.gimbal.update(t)
        if not self.cfg.ablate_prediction:
            self.belief.advance_to(t)
        if t - self.t_loss > self.cfg.max_search:
            self._finish("TIMEOUT", t)
        else:
            if self.phase == "DWELL":
                self._check_dwell_done(t)
            if self.phase == "PLAN":
                t0 = time.perf_counter()
                self.aim = self.choose_aim(t)
                self.compute_ms.append((time.perf_counter() - t0) * 1000)
                self.gimbal.command(self.aim["pitch"], self.aim["yaw"], t)
                self.phase = "SLEW"
            if self.phase == "SLEW" and self.gimbal.ready(t):
                self.phase, self.dwell_start = "DWELL", t
                self.frames, self.saw_det, self.streak = 0, False, 0
            out["cmd"] = (self.gimbal.pitch, self.gimbal.yaw)
        out["events"], self.log_events = self.log_events, []
        if self.state != "SEARCH":
            out["cmd"] = None
        return out

    def _check_dwell_done(self, t):
        elapsed = t - self.dwell_start
        enough = self.frames >= self.cfg.dwell_min_frames and elapsed >= self.cfg.dwell_min_s
        if enough and not self.saw_det:
            self._apply_miss()                                   # a clean look found nothing -> Bayes update
            self.views += 1
            self.phase = "PLAN"
        elif (enough and elapsed >= self.cfg.dwell_max_s) or elapsed >= self.cfg.dwell_max_s + 1.0:
            # Either an ambiguous detection never confirmed, or no frames arrived. Neither is evidence of
            # absence, so NO missed-detection update; just move on.
            self.views += 1
            self.phase = "PLAN"

    def _apply_miss(self):
        if self.strategy != "bayes" or self.cfg.ablate_miss:
            return
        pd = self.cam.detection_probability(self.belief.p[:, :2], self.drone_pos, self.drone_q,
                                            self.gimbal.pitch, self.gimbal.yaw)
        self.belief.missed_detection(pd)

    # ------------------------------------------------------------------ detections
    def on_detections(self, t, dets):
        """
        Every detector frame. dets: list of {'cx', 'cy', 'conf'} (normalised centre, confidence).
        Only frames taken after the gimbal settled count; slewing frames are neither detections nor evidence.
        """
        if self.state != "SEARCH" or self.phase != "DWELL" or self.drone_pos is None:
            return
        if t - self.cfg.frame_latency < self.gimbal.ready_t:
            return
        self.frames += 1
        cands = [d for d in dets if d["conf"] >= self.cfg.conf_min]
        if not cands:
            self.streak = 0
            return
        self.saw_det = True                                      # ambiguous until confirmed: blocks the miss update
        best = max(cands, key=lambda d: d["conf"])
        g = self.cam.pixel_to_ground(best["cx"], best["cy"], self.drone_pos, self.drone_q,
                                     self.gimbal.pitch, self.gimbal.yaw)
        if g is None or not self._in_gate(g, t):
            self.streak = 0
            self.log_events.append(("REJECTED", t, f"candidate outside the motion prior: {g}"))
            return
        if self.last_g is not None and self.streak > 0 and math.hypot(g[0] - self.last_g[0], g[1] - self.last_g[1]) > self.cfg.confirm_radius:
            self.streak = 0                                      # position jumped: not the same object
        self.streak += 1
        self.last_g = g
        if self.streak >= self.cfg.confirm_frames:
            self._finish("REACQUIRED", t, g)

    def _in_gate(self, g, t):
        mean, cov, _ = self.prior(t)
        d = np.array(g) - mean
        return float(d @ np.linalg.inv(cov) @ d) <= self.cfg.gate_chi2

    def _finish(self, outcome, t, found_xy=None):
        ep = {"strategy": self.strategy, "outcome": outcome, "t_loss": self.t_loss, "t_end": t,
              "duration": t - self.t_loss, "views": self.views,
              "gimbal_deg": math.degrees(self.gimbal.moved),
              "compute_ms": float(np.mean(self.compute_ms)) if self.compute_ms else 0.0,
              "found_xy": found_xy}
        self.episodes.append(ep)
        self.last_episode = ep
        self.log_events.append((outcome, t, f"{ep['duration']:.1f}s, {self.views} views"))
        self.state = "TRACK"                                     # the caller releases the gimbal
        self._new_filter()

    # ------------------------------------------------------------------ strategies
    def choose_aim(self, t):
        if self.strategy == "bayes":
            return self._aim_bayes(t)
        if self.strategy == "predict":
            mean, _, _ = self.prior(t)
            return self._aim_at(mean)
        pt = self.queue[self.qi % len(self.queue)]
        self.qi += 1
        return self._aim_at(pt)

    def _aim_at(self, xy, extra=None):
        pitch, yaw = self.cam.aim_angles(xy, self.drone_pos, self.drone_q, self.cfg.pitch_min, self.cfg.pitch_max)
        a = {"pitch": pitch, "yaw": yaw, "aim_xy": (float(xy[0]), float(xy[1]))}
        if extra:
            a.update(extra)
        return a

    def _aim_bayes(self, t):
        """U(a) = sum_i b(i) P_D(i|a) - lambda * tau(a), maximised over aim points drawn from the belief."""
        b = self.belief.grid()
        support = np.argwhere(b >= self.cfg.support_frac * b.max())
        w = b[support[:, 0], support[:, 1]]
        cx, cy = self.belief.cell_centres(support[:, 0], support[:, 1])
        pts = np.column_stack([cx, cy])
        order = np.argsort(-w)
        stride = max(1, len(order) // max(1, self.cfg.n_stride))
        cand = np.unique(np.concatenate([order[:self.cfg.n_top], order[::stride]]))
        best, best_u = None, -1e18
        for k in cand:
            pitch, yaw = self.cam.aim_angles(pts[k], self.drone_pos, self.drone_q, self.cfg.pitch_min, self.cfg.pitch_max)
            pd = self.cam.detection_probability(pts, self.drone_pos, self.drone_q, pitch, yaw)
            gain = float(np.sum(w * pd))                                       # expected probability of seeing it
            tau = self.gimbal.time_to_reach(pitch, yaw)                        # slew + settle
            u = gain - self.cfg.lam * tau
            if u > best_u:
                best_u, best = u, {"pitch": pitch, "yaw": yaw, "aim_xy": (float(pts[k, 0]), float(pts[k, 1])),
                                   "gain": gain, "tau": tau, "utility": u}
        return best

    def _scan_points(self, c):
        """Ground aim points for the baselines, around the last known position c, inside the map."""
        s, R = self.cfg.scan_spacing, self.cfg.baseline_radius
        x0, x1, y0, y1 = self.cfg.bounds
        pts = []
        if self.strategy == "raster":
            ys = np.arange(c[1] - R, c[1] + R + 1e-6, s)
            ys = ys if abs(c[1] - ys[0]) <= abs(c[1] - ys[-1]) else ys[::-1]  # start at the nearer edge
            xs = np.arange(c[0] - R, c[0] + R + 1e-6, s)
            for i, y in enumerate(ys):
                row = xs if i % 2 == 0 else xs[::-1]                           # serpentine
                pts += [(x, y) for x in row]
        else:
            th = 0.0
            while True:
                r = s * th / (2 * math.pi)
                pts.append((c[0] + r * math.cos(th), c[1] + r * math.sin(th)))
                if r > R:
                    break
                th += s / max(r, s)                                            # keep arc length ~ spacing
        pts = [(clamp(x, x0, x1 - 1e-3), clamp(y, y0, y1 - 1e-3)) for x, y in pts]
        return pts