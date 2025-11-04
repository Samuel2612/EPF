import numpy as np
from numpy.random import Generator, PCG64
from scipy.optimize import minimize
from scipy.stats import chi2
from scipy.special import kv, gammaln

class COSStudentTFitter:
    """
    Fit StudentTModel by minimising squared distance between
    empirical and model CF on a COS lattice. Plain class, no typing.
    """

    def __init__(self, model, a, b, N=16, batch_frac=0.02, rng=None):
        self.model = model
        self.a = np.asarray(a, float).reshape(self.model.d)
        self.b = np.asarray(b, float).reshape(self.model.d)
        self.N = int(N)
        self.batch_frac = float(batch_frac)
        self.rng = rng or Generator(PCG64())

        self._factor = np.pi / (self.b - self.a)  # k -> u
        self._total_K = (self.N ** self.model.d)
        self._tau_scalar = 60.0
        self.X_data = None
        self._n_obs = 0

    # ----- lattice utilities -----
    def _k_from_lin_idx(self, idx):
        k_multi = np.array(np.unravel_index(idx, (self.N,) * self.model.d))
        return k_multi.T

    def _u_from_lin_idx(self, idx):
        k = self._k_from_lin_idx(idx)
        return self._factor * k

    # ----- empirical CF -----
    def _empirical_cf_idx(self, idx):
        U = self._u_from_lin_idx(idx)   # (m,d)
        X = self.X_data                 # (n,d)
        UTX = X @ U.T
        return np.mean(np.exp(1j * UTX), axis=0)

    # ----- API -----
    def prepare_empirical_cf(self, X, tau=None):
        self.X_data = np.asarray(X, float)
        self._n_obs = len(self.X_data)
        if tau is not None:
            if np.ndim(tau) == 0:
                self._tau_scalar = float(tau)
            else:
                self._tau_scalar = float(np.mean(np.asarray(tau, float)))

    def _batch_indices(self):
        m = max(8, int(self._total_K * self.batch_frac))
        m = min(m, self._total_K)
        base = self.rng.choice(self._total_K, m, replace=False)

        axis_idx = []
        for i in range(self.model.d):
            coord = [0] * self.model.d
            coord[i] = int(self.rng.integers(self.N))
            axis_idx.append(np.ravel_multi_index(tuple(coord), (self.N,) * self.model.d))

        return np.unique(np.concatenate([base, np.array(axis_idx, dtype=int)]))

    # ----- pack / unpack (Cholesky) -----
    def _pack(self):
        L = np.linalg.cholesky(self.model.Sigma)
        tril = L[np.tril_indices(self.model.d)]
        return np.concatenate([
            self.model.mu,
            self.model.B.ravel(),
            tril,
            np.array([self.model.alpha, self.model.nu], float),
        ])

    def _unpack(self, theta):
        d, k = self.model.d, self.model.k
        t = np.asarray(theta, float)

        self.model.mu = t[:d].copy()
        self.model.B = t[d:d + d * k].reshape(d, k).copy()

        t0 = d + d * k
        t1 = t0 + d * (d + 1) // 2
        tril = t[t0:t1]
        L = np.zeros((d, d))
        L[np.tril_indices(d)] = tril
        self.model.Sigma = L @ L.T + 1e-9 * np.eye(d)

        self.model.alpha = float(t[-2])
        self.model.nu = float(t[-1])

    # ----- loss & fit -----
    def _loss(self, theta):
        self._unpack(theta)
        idx = self._batch_indices()

        phi_emp = self._empirical_cf_idx(idx)

        tauv = np.full(len(idx), self._tau_scalar)
        Z = np.zeros((len(idx), self.model.k))
        u = self._u_from_lin_idx(idx)
        phi_mod = self.model.phi(u, tauv, Z)

        diff = phi_emp - phi_mod
        w = np.exp(-0.04 * np.linalg.norm(u, axis=1))
        return np.sum(w * (diff.real ** 2 + diff.imag ** 2))

    def fit(self, maxiter=400, verbose=False):
        if self.X_data is None:
            raise RuntimeError("Call prepare_empirical_cf(X, tau) before fit().")
        x0 = self._pack()

        d, k = self.model.d, self.model.k
        n_tril = d * (d + 1) // 2
        n_free = d + d * k + n_tril
        bounds = [(-np.inf, np.inf)] * n_free + [(0.05, 3.0), (2.1, 500.0)]

        res = minimize(
            fun=self._loss,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(maxiter=int(maxiter), maxls=20, disp=bool(verbose)),
        )
        self._unpack(res.x)
        self.result_ = res
        return res

    # ----- diagnostics & moments -----
    def diagnostics(self, batch=4000):
        if self.X_data is None:
            raise RuntimeError("Run prepare_empirical_cf first.")
        m = min(int(batch), self._total_K)
        idx = self.rng.choice(self._total_K, m, replace=False)
        u = self._u_from_lin_idx(idx)

        phi_emp = self._empirical_cf_idx(idx)
        tauv = np.full(m, self._tau_scalar)
        Z = np.zeros((m, self.model.k))
        phi_mod = self.model.phi(u, tauv, Z)

        diff = phi_emp - phi_mod
        w = np.exp(-0.04 * np.linalg.norm(u, axis=1))
        Q = np.sum(w * (diff.real ** 2 + diff.imag ** 2))

        p = len(self._pack())
        J = max(self._n_obs, 1) * Q
        pval = 1.0 - chi2.cdf(J, max(1, 2 * m - p))
        cond = float(np.linalg.cond(self.model.Sigma))
        return {"J": float(J), "pval": float(pval), "cond(Sigma)": cond}

    def cos_price_moments(self, n_mom=2):
        k = np.arange(self.N)
        u = np.zeros((self.N, self.model.d))
        u[:, 0] = self._factor[0] * k

        tauv = np.full(self.N, self._tau_scalar)
        Z = np.zeros((self.N, self.model.k))
        phi = self.model.phi(u, tauv, Z)

        Ak = (2.0 / (self.b[0] - self.a[0])) * np.real(np.exp(-1j * u[:, 0] * self.a[0]) * phi)

        moments = np.zeros(int(n_mom), float)
        if n_mom >= 1:
            C0 = (k == 0).astype(float)
            moments[0] = np.sum(Ak * C0)
        if n_mom >= 2:
            moments[1] = np.sum(Ak * (0.5 * (k == 0).astype(float) + np.sinc(k)))
        return moments