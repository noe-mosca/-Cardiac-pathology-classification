"""Classifiers. Every model is a scikit-learn estimator whose first step is a
StandardScaler, so the scaling is fitted on the training folds only.

``mlp_ensemble``  ``n_models`` MLPs (4 hidden layers of 32,
                  BatchNorm, LeakyReLU), each trained on a random 75 % of the
                  patients and 2/3 of the features, probabilities averaged. The
                  models are trained together as one batched network (same
                  maths as separate networks, ~50x faster); dropped features
                  are zeroed, which is equivalent to removing them.
``soft_vote``     alpha * P(mlp) + (1 - alpha) * P(other), alpha chosen by an
                  inner cross-validation on the training folds (never on the
                  evaluation folds).
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


class MLPEnsembleClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_models=50, hidden=32, n_layers=4, epochs=200, lr=1e-3, batch_size=30,
                 subset=0.75, feature_frac=2 / 3, random_state=0):
        self.n_models = n_models
        self.hidden = hidden
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.subset = subset
        self.feature_frac = feature_frac
        self.random_state = random_state

    # --------------------------------------------------------------- network
    def _init_params(self, n_in: int, n_out: int, gen: torch.Generator) -> None:
        dims = [n_in] + [self.hidden] * self.n_layers + [n_out]
        M = self.n_models
        self.weights_, self.biases_ = [], []
        for a, b in zip(dims[:-1], dims[1:]):
            bound = 1 / np.sqrt(a)  # nn.Linear default init
            self.weights_.append(((torch.rand(M, a, b, generator=gen) * 2 - 1) * bound).requires_grad_())
            self.biases_.append(((torch.rand(M, 1, b, generator=gen) * 2 - 1) * bound).requires_grad_())
        self.bn_gamma_ = [torch.ones(M, 1, self.hidden, requires_grad=True) for _ in range(self.n_layers)]
        self.bn_beta_ = [torch.zeros(M, 1, self.hidden, requires_grad=True) for _ in range(self.n_layers)]
        self.bn_mean_ = [torch.zeros(M, 1, self.hidden) for _ in range(self.n_layers)]
        self.bn_var_ = [torch.ones(M, 1, self.hidden) for _ in range(self.n_layers)]

    def _forward(self, x: torch.Tensor, train: bool) -> torch.Tensor:
        """x: (n_models, batch, features) -> logits (n_models, batch, classes)."""
        for i, (w, b) in enumerate(zip(self.weights_, self.biases_)):
            x = torch.baddbmm(b, x, w)
            if i == self.n_layers:
                break
            if train:  # BatchNorm1d, momentum 0.1, per model
                mean, var = x.mean(1, keepdim=True), x.var(1, unbiased=False, keepdim=True)
                n = x.shape[1]
                with torch.no_grad():
                    self.bn_mean_[i].mul_(0.9).add_(0.1 * mean)
                    self.bn_var_[i].mul_(0.9).add_(0.1 * var * n / max(n - 1, 1))
            else:
                mean, var = self.bn_mean_[i], self.bn_var_[i]
            x = (x - mean) / torch.sqrt(var + 1e-5) * self.bn_gamma_[i] + self.bn_beta_[i]
            x = F.leaky_relu(x, 0.01)
        return x

    # --------------------------------------------------------------- sklearn API
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)
        gen = torch.Generator().manual_seed(int(rng.integers(2**31)))
        M, n_sub = self.n_models, int(self.subset * n)
        self.feature_mask_ = np.zeros((M, d), dtype=np.float32)
        for m in range(M):
            self.feature_mask_[m, rng.choice(d, int(self.feature_frac * d), replace=False)] = 1
        patients = np.stack([rng.choice(n, n_sub, replace=False) for _ in range(M)])

        self._init_params(d, len(self.classes_), gen)
        params = self.weights_ + self.biases_ + self.bn_gamma_ + self.bn_beta_
        opt = torch.optim.Adam(params, lr=self.lr)
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y_idx)
        mask = torch.from_numpy(self.feature_mask_).unsqueeze(1)
        rows = torch.arange(M).unsqueeze(1)
        for _ in range(self.epochs):
            order = torch.from_numpy(patients[rows.numpy(), rng.permuted(np.tile(np.arange(n_sub), (M, 1)), axis=1)])
            for s in range(0, n_sub, self.batch_size):
                idx = order[:, s:s + self.batch_size]
                out = self._forward(Xt[idx] * mask, train=True)
                # sum over models of each model's mean loss = independent training
                loss = F.cross_entropy(out.reshape(-1, out.shape[-1]), yt[idx].reshape(-1), reduction="sum") / idx.shape[1]
                opt.zero_grad()
                loss.backward()
                opt.step()
        for p in params:
            p.requires_grad_(False)
        return self

    def predict_proba(self, X):
        X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        x = X.unsqueeze(0).expand(self.n_models, -1, -1) * torch.from_numpy(self.feature_mask_).unsqueeze(1)
        with torch.no_grad():
            return torch.softmax(self._forward(x, train=False), -1).mean(0).numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class SoftVoteClassifier(ClassifierMixin, BaseEstimator):
    """alpha * P(first) + (1 - alpha) * P(second), alpha chosen by inner CV
    (out-of-fold accuracy, ties broken towards 0.5) unless ``alpha`` is fixed."""

    def __init__(self, first, second, alpha=None, alphas=tuple(np.round(np.linspace(0, 1, 11), 2)),
                 inner_splits=5, random_state=0):
        self.first = first
        self.second = second
        self.alpha = alpha
        self.alphas = alphas
        self.inner_splits = inner_splits
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if self.alpha is None:
            cv = StratifiedKFold(self.inner_splits, shuffle=True, random_state=self.random_state)
            p1 = cross_val_predict(clone(self.first), X, y, cv=cv, method="predict_proba")
            p2 = cross_val_predict(clone(self.second), X, y, cv=cv, method="predict_proba")
            self.inner_accuracy_ = {float(a): float(np.mean(self.classes_[(a * p1 + (1 - a) * p2).argmax(1)] == y))
                                    for a in self.alphas}
            best = max(self.inner_accuracy_.values())
            self.alpha_ = min((a for a, s in self.inner_accuracy_.items() if s == best), key=lambda a: abs(a - 0.5))
        else:
            self.alpha_ = float(self.alpha)
        self.first_, self.second_ = clone(self.first).fit(X, y), clone(self.second).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.alpha_ * self.first_.predict_proba(X) + (1 - self.alpha_) * self.second_.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


def _base(name: str, seed: int, **kw):
    if name == "logreg":
        return LogisticRegression(max_iter=5000, **kw)
    if name == "svm":
        return SVC(probability=True, random_state=seed, **kw)
    if name == "svm_linear":
        return SVC(kernel="linear", probability=True, random_state=seed, **kw)
    if name == "svm_rbf":
        return SVC(kernel="rbf", probability=True, random_state=seed, **kw)
    if name == "knn":
        return KNeighborsClassifier(**kw)
    if name == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=1, **kw)
    if name == "extra_trees":
        return ExtraTreesClassifier(random_state=seed, n_jobs=1, **kw)
    if name == "xgboost":
        return XGBClassifier(eval_metric="mlogloss", random_state=seed, n_jobs=1, verbosity=0, **kw)
    if name == "mlp_ensemble":
        return MLPEnsembleClassifier(random_state=seed, **kw)
    raise ValueError(f"unknown model {name!r}")


def build_model(name: str, params: dict | None = None, seed: int = 0):
    """Pipeline(StandardScaler, model). ``soft_vote`` takes ``first`` / ``second``
    as {model: name, params: {...}} and an optional fixed ``alpha``."""
    params = dict(params or {})
    if name == "soft_vote":  # each part is a pipeline with its own scaler
        first, second = (build_model(params[k]["model"], params[k].get("params"), seed) for k in ("first", "second"))
        rest = {k: v for k, v in params.items() if k not in ("first", "second")}
        return SoftVoteClassifier(first, second, random_state=seed, **rest)
    return Pipeline([("scale", StandardScaler()), ("clf", _base(name, seed, **params))])


def grid_for_pipeline(grid: dict | None) -> dict:
    """{C: [...]} -> {clf__C: [...]} for GridSearchCV over a pipeline."""
    return {f"clf__{k}": v for k, v in (grid or {}).items()}
