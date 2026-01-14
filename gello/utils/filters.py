from collections import deque
from typing import Optional, List
import numpy as np
import time

class SignalFilter:
    """
    Lightweight signal filtering suite.
    Supported: 'ema', 'mean', 'butter', 'kalman', 'one_euro'.
    """
    def __init__(
        self, 
        filter_type: str = "none", 
        sampling_rate: float = 125.0,
        # EMA / Mean args
        alpha: float = 0.2, 
        window: int = 5,
        # Butterworth args
        cutoff_hz: float = 10.0,
        # Kalman args (process noise covariance, measurement noise covariance)
        q_process: float = 1e-5,
        r_measure: float = 1e-2,
        # One Euro args
        min_cutoff: float = 1.0,
        beta: float = 0.5,
        d_cutoff: float = 1.0
    ):
        self.type = filter_type.lower()
        self.fs = sampling_rate
        self.dt = 1.0 / sampling_rate
        
        # --- EMA State ---
        self.alpha = alpha
        self.ema_state: Optional[np.ndarray] = None
        
        # --- Mean State ---
        self.window = window
        self.buffer = deque(maxlen=window)

        # --- Butterworth State (2nd order Low Pass) ---
        if self.type == "butter":
            # Direct Difference Equation coefficients for 2nd order Butterworth Lowpass
            t = self.dt
            wc = 2 * np.pi * cutoff_hz
            c = 1.0 / (np.tan(wc * t / 2.0))
            
            self.a0 = 1.0 / (1.0 + np.sqrt(2)*c + c**2)
            self.b0 = 1.0 * self.a0
            self.b1 = 2.0 * self.a0
            self.b2 = 1.0 * self.a0
            self.a1 = 2.0 * (1.0 - c**2) * self.a0
            self.a2 = (1.0 - np.sqrt(2)*c + c**2) * self.a0
            
            self.x_hist = [None, None] # x[n-1], x[n-2]
            self.y_hist = [None, None] # y[n-1], y[n-2]

        # --- Kalman State (1D constant model) ---
        self.q_process = q_process
        self.r_measure = r_measure
        self.P: Optional[np.ndarray] = None # Error covariance
        self.x_est: Optional[np.ndarray] = None # Estimated state

        # --- One Euro State ---
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.one_euro_x: Optional[np.ndarray] = None
        self.one_euro_dx: Optional[np.ndarray] = None
        self.last_time = None

    def _smoothing_factor(self, t_e, cutoff):
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)

    def _low_pass(self, alpha, x, x_prev):
        return alpha * x + (1.0 - alpha) * x_prev

    def update(self, sample: np.ndarray) -> np.ndarray:
        if self.type == "none":
            return sample
        
        # --- EMA ---
        if self.type == "ema":
            if self.ema_state is None:
                self.ema_state = sample
            else:
                self.ema_state = self.alpha * sample + (1.0 - self.alpha) * self.ema_state
            return self.ema_state
            
        # --- Moving Average ---
        if self.type == "mean":
            self.buffer.append(sample)
            if len(self.buffer) == 0:
                return sample
            return np.mean(self.buffer, axis=0)

        # --- Butterworth (2nd Order) ---
        if self.type == "butter":
            if self.x_hist[0] is None:
                # Initialize history with current sample to avoid startup transient
                self.x_hist = [sample, sample]
                self.y_hist = [sample, sample]
                return sample
            
            # Direct Difference Equation:
            # y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
            out = (self.b0 * sample + 
                   self.b1 * self.x_hist[0] + 
                   self.b2 * self.x_hist[1] - 
                   self.a1 * self.y_hist[0] - 
                   self.a2 * self.y_hist[1])
            
            # Shift history
            self.x_hist[1] = self.x_hist[0]
            self.x_hist[0] = sample
            self.y_hist[1] = self.y_hist[0]
            self.y_hist[0] = out
            return out

        # --- Kalman Filter (Simple 1D) ---
        if self.type == "kalman":
            # Initialize
            if self.x_est is None:
                self.x_est = sample
                self.P = np.ones_like(sample) * 1.0
                return sample
            
            # Predict (Constant model: x_k = x_k-1)
            # x_pred = x_est
            # P_pred = P + Q
            P_pred = self.P + self.q_process
            
            # Update
            # K = P_pred / (P_pred + R)
            K = P_pred / (P_pred + self.r_measure)
            
            # x_est = x_pred + K * (z - x_pred)
            self.x_est = self.x_est + K * (sample - self.x_est)
            
            # P = (1 - K) * P_pred
            self.P = (1.0 - K) * P_pred
            
            return self.x_est

        # --- One Euro Filter ---
        if self.type == "one_euro":
            now = time.time()
            if self.last_time is None:
                dt = self.dt # Use default dt on first step
            else:
                dt = now - self.last_time
            self.last_time = now

            if self.one_euro_x is None:
                self.one_euro_x = sample
                self.one_euro_dx = np.zeros_like(sample)
                return sample

            # Filter derivative
            dx_raw = (sample - self.one_euro_x) / dt
            alpha_d = self._smoothing_factor(dt, self.d_cutoff)
            self.one_euro_dx = self._low_pass(alpha_d, dx_raw, self.one_euro_dx)

            # Filter signal
            # Use derivative magnitude to tune cutoff
            # The faster we move, the higher the cutoff (less filtering)
            cutoff = self.min_cutoff + self.beta * np.abs(self.one_euro_dx)
            alpha = self._smoothing_factor(dt, cutoff)
            self.one_euro_x = self._low_pass(alpha, sample, self.one_euro_x)
            
            return self.one_euro_x

        return sample
