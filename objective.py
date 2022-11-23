from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Objective(BaseObjective):
    name = "Huber Regression with L2 regularization"

    parameters = {
        'epsilon': [1.35],
        'lmbd': [1.],
    }

    def __init__(self, epsilon=1.35, lmbd=1.):
        self.epsilon = epsilon
        self.lmbd = lmbd

    def set_data(self, X, y):
        self.X, self.y = X, y

    def compute(self, params):
        intercept, scale, coef = params[0], params[1], params[2:]

        z = np.abs(self.y - self.X @ coef - intercept) / scale
        outlier_mask = z > self.epsilon
        n_samples = len(self.X)
        loss = np.empty(n_samples)
        loss[~outlier_mask] = scale + z[~outlier_mask]**2 * scale
        loss[outlier_mask] = scale + (
            2 * self.epsilon * z[outlier_mask] - self.epsilon**2) * scale
        return loss.sum() + self.lmbd * np.dot(coef, coef)

    def get_objective(self):
        return dict(X=self.X, y=self.y, lmbd=self.lmbd, epsilon=self.epsilon)
