"""
belief_grid.py

Divides the map into square cells and assigns each cell the probability
that the lost target is inside it. The probabilities come from the Kalman
filter's last valid state, predicted forward in time to "now".
"""

import numpy as np 
from target_kalman_filter import TargetKalmanFilter  # to use the filter's predict() maths

# MAP CONFIGURATION 
X_MIN = 0.0     # Western edge of the map in ENU metres
X_MAX = 1556.42  # Eastern edge of the map in ENU metres
Y_MIN = 0.0     # Southern edge of the map in ENU metres
Y_MAX = 1576.91  # Northern edge of the map in ENU metres

CELL_SIZE = 10.0    # ADJUST, currently matches the ~11 m Kalman error

POSITION_STD_FLOOR = 10.0  # minimum position uncertainty (metres), 0.0 will disable it


class BeliefGrid:
    """Holds one probability value per map cell, summing to 1."""

    def __init__(self, x_min, x_max, y_min, y_max, cell_size, position_std_floor=0.0):
        self.x_min = x_min                          # store the map's western edge
        self.y_min = y_min                          # store the map's southern edge
        self.cell_size = cell_size                  # store the cell width/height in metres
        self.position_std_floor = position_std_floor  # store the minimum position uncertainty

        self.nx = int(np.ceil((x_max - x_min) / cell_size))  # number of cells east to west (rounded up so the map is fully covered)
        self.ny = int(np.ceil((y_max - y_min) / cell_size))  # number of cells north to south

        self.x_centres = x_min + (np.arange(self.nx) + 0.5) * cell_size  # east coordinate of every column's CENTRE
        self.y_centres = y_min + (np.arange(self.ny) + 0.5) * cell_size  # north coordinate of every row's CENTRE

        self.X, self.Y = np.meshgrid(self.x_centres, self.y_centres, indexing="ij")  # 2D arrays: the (x, y) of every cell centre, shape (nx, ny)

        self.belief = np.full((self.nx, self.ny), 1.0 / (self.nx * self.ny))  # Start as equal probability in each cell before information is available

        self.snap_x = None               # snapshot of the filter's state vector [x, y, vx, vy]
        self.snap_P = None               # snapshot of the filter's 4x4 covariance matrix P
        self.snap_t = None               # timestamp of the last valid observation t
        self.process_noise_std = None    # the filter's process noise setting, so that prediction matches the filter

    def set_snapshot(self, kf):
        """Freeze a copy of the filter's state at the moment of last valid observation."""
        state = kf.get_state()  # returns (x, P) copies, or None if the filter never initialised
        if state is None:  # If there is nothing to freeze
            raise ValueError("Filter has no state yet, nothing to snapshot.")  # Leaves a note that failure has occurred instead of building a random belief
        self.snap_x, self.snap_P = state  # store the frozen state and covariance
        self.snap_t = kf.last_timestamp   # store when that state was last confirmed by a detection
        self.process_noise_std = kf.process_noise_std  # store the noise setting used by the filter

    def belief_at(self, t_now):
        """Predict the snapshot forward to t_now and fill the grid with the result."""
        if self.snap_x is None:  # If no snapshot has been stored
            raise RuntimeError("Call set_snapshot() first.")  # because the prediction needs a starting point
        dt = t_now - self.snap_t  # how many seconds since the last valid observation
        if dt < 0:  # because asking for a time before the snapshot makes no sense (only need after)
            raise ValueError("t_now is earlier than the snapshot time.")  # Fail note if this happens

        predictor = TargetKalmanFilter(process_noise_std=self.process_noise_std)  # a filter used only for its predict() maths
        predictor.x = self.snap_x.copy()  # load the frozen state 
        predictor.P = self.snap_P.copy()  # load the frozen covariance
        predictor.last_timestamp = self.snap_t  # load the snapshot time
        if dt > 0:  # only predict if time has actually passed
            predictor.predict(dt)  # apply F and Q from Kalman filter, which moves the mean along the velocity and grows the uncertainty

        mean = predictor.x[0:2]  # predicted position [x, y]: (dont need velocity here, only position matters for cells)
        cov = predictor.P[0:2, 0:2] + (self.position_std_floor ** 2) * np.eye(2)  # position block of P

        x_max_eff = self.x_min + self.nx * self.cell_size  # actual eastern edge after rounding up
        y_max_eff = self.y_min + self.ny * self.cell_size  # actual northern edge after rounding up
        if not (self.x_min <= mean[0] < x_max_eff and self.y_min <= mean[1] < y_max_eff):  # predicted mean has left the map
            print("WARNING: predicted target position is outside the grid.")  # the belief will pile up on the map edge and cause an error

        cov_inv = np.linalg.inv(cov)  # inverse covariance, needed by the Gaussian formula
        dx = self.X - mean[0]  # east offset of every cell centre from the predicted mean
        dy = self.Y - mean[1]  # north offset of every cell centre from the predicted mean
        exponent = -0.5 * (cov_inv[0, 0] * dx * dx + 2 * cov_inv[0, 1] * dx * dy + cov_inv[1, 1] * dy * dy)  # Gaussian exponent (in quadratic form) for every cell
        exponent -= exponent.max()  # shift so the best cell has exponent 0, preventing numerical underflow
        b = np.exp(exponent)  # unnormalised Gaussian value in every cell
        b /= b.sum()  # normalise so all cells sum to exactly 1
        self.belief = b  # store the result as the current belief
        return mean, cov  # also return the predicted mean and covariance for inspection

    def most_likely_cell(self):
        """Return (x, y, probability) of the single most probable cell."""
        ix, iy = np.unravel_index(np.argmax(self.belief), self.belief.shape)  # grid index of the largest value
        return self.x_centres[ix], self.y_centres[iy], self.belief[ix, iy]  # that cell's centre and its probability

    def cell_index(self, x, y):
        """Convert world coordinates to grid indices, or None if off the map."""
        ix = int((x - self.x_min) // self.cell_size)  # column containing x
        iy = int((y - self.y_min) // self.cell_size)  # row containing y
        if 0 <= ix < self.nx and 0 <= iy < self.ny:  # inside the grid
            return ix, iy  # valid indices
        return None  # outside the map