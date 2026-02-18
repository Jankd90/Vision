import cv2
import numpy as np


class TipKalman2D:
    def __init__(self, dt: float = 1 / 30, process_var: float = 1.0, meas_var: float = 25.0):
        self.kf = cv2.KalmanFilter(4, 2)

        self.kf.transitionMatrix = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float32,
        )

        self.kf.processNoiseCov = process_var * np.eye(4, dtype=np.float32)
        self.kf.measurementNoiseCov = meas_var * np.eye(2, dtype=np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        self.initialized = False

    def init(self, x: float, y: float) -> None:
        self.kf.statePost = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
        self.initialized = True

    def predict(self) -> tuple[float, float]:
        pred = self.kf.predict()
        return float(pred[0, 0]), float(pred[1, 0])

    def update(self, x: float, y: float) -> tuple[float, float]:
        meas = np.array([[x], [y]], dtype=np.float32)
        est = self.kf.correct(meas)
        return float(est[0, 0]), float(est[1, 0])
