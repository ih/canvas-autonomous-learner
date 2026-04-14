"""Verifier driven by a fake Hardware — no real robot, no real CWM.

Exercises the predict -> execute -> observe -> MSE flow and confirms the
reported MSE matches a hand-computed value.
"""

import numpy as np

from learner.verifier import verify_once, quantize_motor


class FakeHardware:
    def __init__(self, pred_base, pred_wrist, actual_base, actual_wrist):
        self.pred_base = pred_base
        self.pred_wrist = pred_wrist
        self.actual_base = actual_base
        self.actual_wrist = actual_wrist
        self.executed = []
        self._first_observe = True

    def observe(self):
        if self._first_observe:
            self._first_observe = False
            cams = {"base": self.pred_base.copy(), "wrist": self.pred_wrist.copy()}
        else:
            cams = {"base": self.actual_base.copy(), "wrist": self.actual_wrist.copy()}
        motor = np.zeros(6, dtype=np.float32)
        ctx = np.zeros((448, 224, 3), dtype=np.uint8)
        return cams, motor, ctx

    def predict(self, ctx, motor, action):
        return self.pred_base, self.pred_wrist

    def execute(self, action):
        self.executed.append(action)


def test_verify_zero_mse_when_prediction_matches():
    frame = np.full((224, 224, 3), 128, dtype=np.uint8)
    hw = FakeHardware(frame, frame, frame, frame)
    probe = verify_once(hw, action=1, settle_time=0.0)
    assert probe.mse == 0.0
    assert probe.action == 1
    assert hw.executed == [1]


def test_verify_nonzero_mse_on_mismatch():
    pred = np.zeros((224, 224, 3), dtype=np.uint8)
    actual = np.full((224, 224, 3), 255, dtype=np.uint8)
    hw = FakeHardware(pred, pred, actual, actual)
    probe = verify_once(hw, action=2, settle_time=0.0)
    # normalized to [0,1], diff is 1.0 everywhere -> MSE = 1.0
    assert abs(probe.mse - 1.0) < 1e-6


def test_quantize_motor_stable():
    a = np.array([0.1, 4.0, -3.0, 11.0, -13.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 3.5, -4.0, 10.8, -12.9, 0.4], dtype=np.float32)
    # bin_size=10 buckets these all into [0, 0, 0, 1, -1, 0]
    assert quantize_motor(a) == quantize_motor(b)
    assert quantize_motor(a) == "0,0,0,1,-1,0"
