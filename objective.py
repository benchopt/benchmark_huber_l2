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
        coef, intercept = params[1:], params[0]
        z = self.y - self.X @ coef - intercept
        outlier_mask = np.abs(z) < self.epsilon
        loss = np.sum(z[~outlier_mask] ** 2)
        loss += np.sum(self.epsilon * (z[outlier_mask] ** 2) - self.epsilon ** 2)
        return loss + self.lmbd * np.dot(coef, coef)

    def to_dict(self):
        return dict(X=self.X, y=self.y, lmbd=self.lmbd, epsilon=self.epsilon)
