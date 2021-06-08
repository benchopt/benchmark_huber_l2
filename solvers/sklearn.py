import warnings
from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.linear_model import HuberRegressor
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    name = 'sklearn'

    install_cmd = 'conda'
    requirements = ['scikit-learn']
    references = [
        'F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, '
        'O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, '
        'J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot'
        ' and E. Duchesnay'
        '"Scikit-learn: Machine Learning in Python", J. Mach. Learn. Res., '
        'vol. 12, pp. 2825-283 (2011)'
    ]

    def set_objective(self, X, y, lmbd, epsilon):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.clf = HuberRegressor(
            alpha=self.lmbd, fit_intercept=True, epsilon=epsilon, tol=0
        )
        warnings.filterwarnings('ignore', category=ConvergenceWarning)

    def run(self, n_iter):
        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.y)

    def get_result(self):
        return np.r_[self.clf.intercept_, self.clf.coef_]
