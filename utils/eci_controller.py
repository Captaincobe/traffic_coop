import numpy as np

class ECIThresholdController:
    def __init__(self, initial_threshold=0.5, learning_rate=0.01, target_coverage=0.9):
        self.q_t = initial_threshold
        self.eta = learning_rate
        self.alpha = 1 - target_coverage
        self.coverage_window = []
        self.window_size = 100

    def update(self, nonconformity_score, covered):
        """更新阈值 q_t（类似ECI式的反馈机制）"""
        err = int(not covered)  # 0 if covered, 1 if not
        smooth_term = (nonconformity_score - self.q_t) * self._sigmoid_grad(nonconformity_score - self.q_t)
        delta = self.eta * (err - self.alpha + smooth_term)
        self.q_t += delta
        self.coverage_window.append(1 - err)
        if len(self.coverage_window) > self.window_size:
            self.coverage_window.pop(0)

    def _sigmoid_grad(self, x, c=1.0):
        s = 1 / (1 + np.exp(-c * x))
        return c * s * (1 - s)

    def is_in_conformal_set(self, scores):
        """判断当前输入是否在 conformal 集合内"""
        return np.max(scores) >= self.q_t

    def get_threshold(self):
        return self.q_t

    def get_coverage(self):
        if not self.coverage_window:
            return None
        return np.mean(self.coverage_window)
    

