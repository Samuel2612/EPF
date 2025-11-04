import numpy as np
from scipy import special, optimize  # pip install scipy

class test_mvt_ll:
    """
    Fast, stable MLE for multivariate Student-t via ECM.
    Key fix: update nu by 1D *bounded maximization* of the observed log-likelihood.
    """

    def __init__(self, n=5000, p=5, nu_true=7.0, seed=42):
        self.n = int(n); self.p = int(p); self.nu_true = float(nu_true)
        self.rng = np.random.default_rng(seed)

        # True parameters (scale Σ_true; covariance = nu/(nu-2)*Σ_true)
        self.mu_true = np.linspace(100.0,100.4, self.p)
        A = self.rng.standard_normal((self.p, self.p))
        S = A @ A.T
        S = S / np.sqrt(np.outer(np.diag(S), np.diag(S)))  
        S = 0.5*(S + S.T) - np.diag(np.linspace(0.5, 1.5, self.p))/10

        self.Sigma_true = S
        print(self.Sigma_true)


    def _sample_mvt(self):
        L = np.linalg.cholesky(self.Sigma_true)
        Z = self.rng.standard_normal((self.n, self.p))
        U = self.rng.chisquare(df=self.nu_true, size=self.n)     
        W = np.sqrt(self.nu_true / U)[:, None]                    #
        return self.mu_true + (Z @ L.T) * W


    @staticmethod
    def _mahalanobis2(X, mu, Sigma):
        L = np.linalg.cholesky(Sigma)
        Y = np.linalg.solve(L, (X - mu).T)    # p × n
        return np.sum(Y*Y, axis=0)            # length n

    @staticmethod
    def _logdet(Sigma):
        return 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(Sigma))))


    def _e_step(self, X, mu, Sigma, nu):
        d2 = self._mahalanobis2(X, mu, Sigma)
        w = (nu + self.p) / (nu + d2)  # E[λ | x]
        return w, d2

    def _m_mu(self, X, w):
        sw = np.sum(w)
        return (w[:, None] * X).sum(axis=0) / sw

    def _m_Sigma(self, X, mu, w, ridge=1e-8):
        D = X - mu
        WD = D * w[:, None]
        S = (WD.T @ D) / X.shape[0]   # (1/n) Σ w_i (x-μ)(x-μ)^T  → scale matrix
        tr = float(np.trace(S)) or 1.0
        return 0.5*(S + S.T) + ridge*(tr/self.p)*np.eye(self.p)

    # ν-dependent part of the observed log-likelihood (μ,Σ fixed via d2)
    def _ll_nu_given_d2(self, nu, d2):
        p, n = self.p, d2.size
        # constant terms in μ,Σ are omitted (not needed for argmax w.r.t. ν)
        return (
            n * (special.gammaln(0.5*(nu+p)) - special.gammaln(0.5*nu) - 0.5*p*np.log(nu))
            - 0.5 * (nu + p) * np.sum(np.log1p(d2 / nu))
        )

    def _update_nu_brent(self, d2, nu_min=0.5001, nu_max=500.0):
        # maximize _ll_nu_given_d2 with respect to ν on [nu_min, nu_max]
        # use negative for minimize_scalar (bounded Brent)
        def obj(nu):
            return -self._ll_nu_given_d2(nu, d2)
        res = optimize.minimize_scalar(obj, bounds=(nu_min, nu_max), method='bounded', options={'xatol':1e-6, 'maxiter':200})
        return float(res.x)

    def _loglik_full(self, X, mu, Sigma, nu):
        n, p = X.shape
        d2 = self._mahalanobis2(X, mu, Sigma)
        return (
            n * (special.gammaln(0.5*(nu+p)) - special.gammaln(0.5*nu) - 0.5*p*np.log(nu*np.pi))
            - 0.5 * n * self._logdet(Sigma)
            - 0.5 * (nu + p) * np.sum(np.log1p(d2 / nu))
        )

    def fit(self, X, max_iter=200, tol=1e-6, verbose=False):
        n, p = X.shape
        mu = X.mean(axis=0)
        S = np.cov(X, rowvar=False, ddof=1)
        tr = float(np.trace(S)) or 1.0
        Sigma = 0.5*(S + S.T) + 1e-8*(tr/p)*np.eye(p)
        
        try:
            
            S_inv = np.linalg.pinv(S)
            Xc = X - mu
            d2 = np.einsum('ni,ij,nj->n', Xc, S_inv, Xc)
            k = np.mean(d2**2) / (p*(p+2))
            nu = max((4*k - 2)/(k - 1), 4.1)
        except Exception:
            nu = 10.0

        ll_prev = self._loglik_full(X, mu, Sigma, nu)
        for it in range(1, max_iter+1):
            w, d2 = self._e_step(X, mu, Sigma, nu)
            mu = self._m_mu(X, w)
            Sigma = self._m_Sigma(X, mu, w)
            # recompute d2 at updated (μ,Σ) for the ν step (ECM)
            d2 = self._mahalanobis2(X, mu, Sigma)
            nu = self._update_nu_brent(d2)

            ll = self._loglik_full(X, mu, Sigma, nu)
            if verbose and (it <= 3 or it % 10 == 0):
                print(f"it={it:3d}  ll={ll:.3f}  nu={nu:.4f}")
            if abs(ll - ll_prev) / (1.0 + abs(ll_prev)) < tol:
                break
            ll_prev = ll

        return mu, Sigma, nu, ll

    def run(self, verbose=False):
        X = self._sample_mvt()
        mu_hat, Sigma_hat, nu_hat, ll = self.fit(X, verbose=verbose)

        # diagnostics
        mu_err = np.linalg.norm(mu_hat - self.mu_true)
        Sigma_err = np.linalg.norm(Sigma_hat - self.Sigma_true, ord='fro')
        cov_true = self.Sigma_true * self.nu_true / (self.nu_true - 2.0)
        cov_hat  = Sigma_hat  * nu_hat       / (nu_hat - 2.0)
        cov_err  = np.linalg.norm(cov_hat - cov_true, ord='fro')

        print(Sigma_hat)
        print(f"n={self.n}, p={self.p} | nu_true={self.nu_true:.3f} → nu_hat={nu_hat:.3f}")
        print(f"||mu_hat - mu_true||_2 = {mu_err:.4e}")
        print(f"||Sigma_hat - Sigma_true||_F = {Sigma_err:.4e}  (scale)")
        print(f"||Cov_hat - Cov_true||_F    = {cov_err:.4e}  (derived)")
        return {
            "mu_hat": mu_hat, "Sigma_hat": Sigma_hat, "nu_hat": nu_hat, "loglik": ll,
            "mu_true": self.mu_true, "Sigma_true": self.Sigma_true, "nu_true": self.nu_true,
            "errors": {"mu_l2": float(mu_err), "Sigma_fro": float(Sigma_err), "Cov_fro": float(cov_err)}
        }

# --- quick demo ---
if __name__ == "__main__":
    t = test_mvt_ll(n=12*90, p=5, nu_true=20, seed=18)
    _ = t.run(verbose=False)
