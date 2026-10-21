"""
search_belief.py

The lost-target belief. The probability of the target being in each map cell is carried by
particles (samples of the full state [x, y, vx, vy]); the grid is simply their histogram.

Why particles: once the camera looks somewhere and sees nothing, the belief stops being a Gaussian.
Particles stay exact through that - propagate each one with the motion model, down-weight the ones
the camera could have seen - and need nothing beyond numpy.

  start():             sample b_0 = N(x_last, P_last), plus the position floor
  advance_to(t):       predict forward with the constant-velocity model + process noise
  missed_detection():  Bayes update  w <- w * (1 - P_D), then renormalise
"""

import numpy as np

from belief_grid import X_MIN, X_MAX, Y_MIN, Y_MAX, CELL_SIZE, POSITION_STD_FLOOR


class SearchBelief:
    def __init__(self, n_particles=4000, process_noise_std=0.5, pos_floor=POSITION_STD_FLOOR,
                 bounds=(X_MIN, X_MAX, Y_MIN, Y_MAX), cell_size=CELL_SIZE, seed=None):
        self.n = n_particles
        self.q = process_noise_std              # same meaning as the Kalman filter's process_noise_std
        self.pos_floor = pos_floor              # extra position uncertainty (stopgap for an overconfident P)
        self.x_min, self.x_max, self.y_min, self.y_max = bounds
        self.cell = cell_size
        self.nx = int(np.ceil((self.x_max - self.x_min) / cell_size))
        self.ny = int(np.ceil((self.y_max - self.y_min) / cell_size))
        self.x_edges = self.x_min + cell_size * np.arange(self.nx + 1)
        self.y_edges = self.y_min + cell_size * np.arange(self.ny + 1)
        self.rng = np.random.default_rng(seed)
        self.p = None                           # (N, 4) particle states
        self.w = None                           # (N,) weights, sum to 1
        self.t = None                           # time the particles currently describe

    def start(self, x_last, P_last, t_last):
        """Initialise from the filter's last valid state (the supervisor's b_0)."""
        self.p = self.rng.multivariate_normal(x_last, P_last, size=self.n, check_valid="ignore")
        self.p[:, :2] += self.rng.normal(0.0, self.pos_floor, (self.n, 2))
        self.w = np.full(self.n, 1.0 / self.n)
        self.t = t_last

    def advance_to(self, t):
        """Constant-velocity prediction with white-noise acceleration - the same model as the Kalman filter's Q."""
        dt = t - self.t
        if dt <= 0:
            return
        a = self.rng.normal(0.0, self.q, (self.n, 2))
        self.p[:, 0:2] += self.p[:, 2:4] * dt + 0.5 * a * dt * dt
        self.p[:, 2:4] += a * dt
        self.t = t
        out = ((self.p[:, 0] < self.x_min) | (self.p[:, 0] >= self.x_max) |
               (self.p[:, 1] < self.y_min) | (self.p[:, 1] >= self.y_max))
        self.w[out] = 0.0                       # a target that left the map is no longer in the search area
        self._normalise()

    def missed_detection(self, pd):
        """Valid image, no detection: lower the weight of every particle the camera would probably have seen."""
        self.w *= (1.0 - pd)
        self._normalise()
        if 1.0 / np.sum(self.w ** 2) < self.n / 2:          # effective sample size too low -> resample
            self._resample()

    def _normalise(self):
        s = self.w.sum()
        if s <= 1e-300:                         # everything was ruled out - fall back to the unweighted cloud
            self.w[:] = 1.0 / self.n
        else:
            self.w /= s

    def _resample(self):
        """Systematic resampling plus a little jitter so duplicated particles spread out again."""
        pos = (self.rng.random() + np.arange(self.n)) / self.n
        idx = np.minimum(np.searchsorted(np.cumsum(self.w), pos), self.n - 1)
        self.p = self.p[idx] + self.rng.normal(0.0, [0.5, 0.5, 0.05, 0.05], (self.n, 4))
        self.w = np.full(self.n, 1.0 / self.n)

    def grid(self):
        """Probability per cell, shape (nx, ny), x index first (east), summing to 1."""
        h, _, _ = np.histogram2d(self.p[:, 0], self.p[:, 1], bins=[self.x_edges, self.y_edges], weights=self.w)
        return h

    def cell_centres(self, ix, iy):
        return self.x_min + (ix + 0.5) * self.cell, self.y_min + (iy + 0.5) * self.cell

    def peak(self):
        g = self.grid()
        ix, iy = np.unravel_index(np.argmax(g), g.shape)
        x, y = self.cell_centres(ix, iy)
        return float(x), float(y), float(g[ix, iy])

    def entropy(self):
        g = self.grid().ravel()
        g = g[g > 0]
        return float(-(g * np.log(g)).sum())


if __name__ == "__main__":
    sb = SearchBelief(seed=1)
    sb.start(np.array([800.0, 700.0, 0.0, 5.0]), np.diag([25.0, 25.0, 2.0, 2.0]), 0.0)
    sb.advance_to(10.0)
    print("after 10 s: peak", [round(v, 3) for v in sb.peak()], "entropy", round(sb.entropy(), 2), "sum", round(sb.grid().sum(), 6))
    # pretend the camera looked at a patch around (800, 750) and saw nothing
    pd = np.where(np.hypot(sb.p[:, 0] - 800, sb.p[:, 1] - 750) < 40, 0.9, 0.0)
    before = sb.w[pd > 0].sum()
    sb.missed_detection(pd)
    print(f"mass inside the looked-at patch: {before:.3f} -> {sb.w[pd > 0].sum():.3f}; total {sb.w.sum():.6f}")
    print("peak moved to", [round(v, 3) for v in sb.peak()])