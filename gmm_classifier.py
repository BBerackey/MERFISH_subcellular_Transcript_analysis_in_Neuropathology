import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture


class GMMClassifier(BaseEstimator, ClassifierMixin):
    """Class-conditional Gaussian Mixture Model classifier.

    One GaussianMixture is fit per class (e.g. per Leiden community) to model p(x | class).
    Prediction uses Bayes' rule:  p(class | x) ∝ p(x | class) * p(class)

    Parameters
    ----------
    n_components : int or 'bic'
        Mixture components per class. 'bic' picks the best k in `component_range` per class by BIC.
    component_range : iterable of int
        Candidate k values when n_components='bic'.
    covariance_type : {'full', 'diag', 'tied', 'spherical'}
    priors : None | 'uniform' | dict/array
        None -> empirical class frequencies; 'uniform' -> equal priors (useful when the training
        set is downsampled and community sizes are not representative).
    reg_covar : float
        Added to covariance diagonals (stabilizes sparse / near-constant count features).
    """

    def __init__(self, n_components=3, component_range=range(1, 7), covariance_type='full',
                 priors=None, reg_covar=1e-4, max_iter=300, n_init=2, random_state=0):
        self.n_components = n_components
        self.component_range = component_range
        self.covariance_type = covariance_type
        self.priors = priors
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

    def _make_gmm(self, k):
        return GaussianMixture(n_components=k, covariance_type=self.covariance_type,
                               reg_covar=self.reg_covar, max_iter=self.max_iter,
                               n_init=self.n_init, random_state=self.random_state)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_, counts = np.unique(y, return_counts=True)
        self.gmms_, self.n_components_ = [], []
        for c in self.classes_:
            Xc = X[y == c]
            if self.n_components == 'bic':
                ks = [k for k in self.component_range if k <= len(Xc)]
                fits = [self._make_gmm(k).fit(Xc) for k in ks]
                best = fits[int(np.argmin([g.bic(Xc) for g in fits]))]
            else:
                best = self._make_gmm(min(self.n_components, len(Xc))).fit(Xc)
            self.gmms_.append(best)
            self.n_components_.append(best.n_components)

        if self.priors is None:
            p = counts / counts.sum()
        elif isinstance(self.priors, str) and self.priors == 'uniform':
            p = np.full(len(self.classes_), 1.0 / len(self.classes_))
        elif isinstance(self.priors, dict):
            p = np.array([self.priors[c] for c in self.classes_], dtype=float)
            p = p / p.sum()
        else:
            p = np.asarray(self.priors, dtype=float)
            p = p / p.sum()
        self.log_priors_ = np.log(p)
        return self

    def _joint_log_likelihood(self, X, batch_size=200_000):
        X = np.asarray(X, dtype=np.float64)
        out = np.empty((X.shape[0], len(self.classes_)))
        for s in range(0, X.shape[0], batch_size):  # chunked: full datasets can be millions of cells
            xb = X[s:s + batch_size]
            for j, g in enumerate(self.gmms_):
                out[s:s + batch_size, j] = g.score_samples(xb) + self.log_priors_[j]
        return out

    def predict_log_proba(self, X):
        jll = self._joint_log_likelihood(X)
        return jll - logsumexp(jll, axis=1, keepdims=True)

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]

    def max_log_likelihood(self, X, batch_size=200_000):
        """max_c log p(x | c): how well the best-fitting class explains x (novelty score; low = outlier)."""
        X = np.asarray(X, dtype=np.float64)
        out = np.full(X.shape[0], -np.inf)
        for s in range(0, X.shape[0], batch_size):
            xb = X[s:s + batch_size]
            for g in self.gmms_:
                out[s:s + batch_size] = np.maximum(out[s:s + batch_size], g.score_samples(xb))
        return out

    def loglik_threshold(self, X_train, quantile=0.01):
        """Novelty cutoff: the `quantile` of max log-likelihood over (training) cells."""
        return float(np.quantile(self.max_log_likelihood(X_train), quantile))

    def predict_with_confidence(self, X, min_proba=None, min_loglik=None, unassigned_label='unassigned'):
        """Return (labels, max posterior, max class log-likelihood).

        Cells whose max posterior < `min_proba` (ambiguous between classes) or whose best
        class log-likelihood < `min_loglik` (fits no class; out-of-distribution) get `unassigned_label`.
        """
        jll = self._joint_log_likelihood(X)
        proba = np.exp(jll - logsumexp(jll, axis=1, keepdims=True))
        conf = proba.max(axis=1)
        loglik = (jll - self.log_priors_).max(axis=1)
        labels = self.classes_[proba.argmax(axis=1)].astype(object)
        if min_proba is not None:
            labels[conf < min_proba] = unassigned_label
        if min_loglik is not None:
            labels[loglik < min_loglik] = unassigned_label
        return labels, conf, loglik
