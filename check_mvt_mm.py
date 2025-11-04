import numpy as np
import pandas as pd
from check_mvt_chf import  MVTCFEstimator

class test_mvt_mm:
    """
    Minimal test: generate 5D multivariate-t samples
    """

    def __init__(self, n=5000, p=5, nu_true=7.0, seed=0):
        self.n = int(n)
        self.p = int(p)
        self.nu_true = float(nu_true)
        self.rng = np.random.default_rng(seed)

        self.mu_true = np.linspace(0.0, 0.4, self.p)  
        A = self.rng.standard_normal((self.p, self.p))
        Sigma_true = A @ A.T


        Sigma_true = (Sigma_true / np.sqrt(np.outer(np.diag(Sigma_true), np.diag(Sigma_true))))
        Sigma_true = (Sigma_true + Sigma_true.T) / 2
        Sigma_true -= np.diag(np.linspace(0.5, 1.5, self.p))/10
        self.Sigma_true = Sigma_true  # t 'scale' (not covariance)
        print(self.Sigma_true)

 
    def _sample_mvt(self):
        L = np.linalg.cholesky(self.Sigma_true)
        Z = self.rng.standard_normal((self.n, self.p))
        U = self.rng.chisquare(df=self.nu_true, size=self.n)
        W = np.sqrt(self.nu_true / U)[:, None]
        X = self.mu_true + (Z @ L.T) * W
        return X

    @staticmethod
    def _mardia_kurtosis(X):
        mu = X.mean(axis=0, keepdims=True)
        Xc = X - mu
        S = np.cov(X, rowvar=False, ddof=1)
        S_inv = np.linalg.pinv(S)
        d2 = np.einsum("ni,ij,nj->n", Xc, S_inv, Xc)
        return float(np.mean(d2**2)), S

    @staticmethod
    def _nu_from_mardia(beta_2p, p, eps=1e-9):
        k = beta_2p / (p * (p + 2))
        if k <= 1 + eps:
            return float(np.inf)
        nu_hat = (4.0 * k - 2.0) / (k - 1.0)
        return float(max(nu_hat, 4.0 + 1e-6))

    def run(self):
        X = self._sample_mvt()

  
        mu_hat = X.mean(axis=0)
        beta_2p, S_hat = self._mardia_kurtosis(X)
        nu_hat = self._nu_from_mardia(beta_2p, self.p)

        # Estimate t 'scale' Sigma (since Cov = nu/(nu-2) * Sigma ⇒ Sigma = Cov * (nu-2)/nu)
        if np.isfinite(nu_hat) and nu_hat > 2:
            Sigma_hat = S_hat * (nu_hat - 2.0) / nu_hat
        else:
            Sigma_hat = S_hat.copy()  # fallback; won't be used if nu_hat<=2


        mu_err = np.linalg.norm(mu_hat - self.mu_true)
        cov_true = self.Sigma_true * self.nu_true / (self.nu_true - 2.0)  # true covariance
        cov_err_frob = np.linalg.norm(S_hat - cov_true, ord='fro')
        sigma_err_frob = np.linalg.norm(Sigma_hat - self.Sigma_true, ord='fro')
        
        print(Sigma_hat)

        results = {
            "mu_hat": mu_hat,
            "S_hat_cov": S_hat,             
            "beta_2p": beta_2p,
            "nu_hat": nu_hat,
            "Sigma_hat": Sigma_hat,           # estimated t scale
            "mu_true": self.mu_true,
            "Sigma_true": self.Sigma_true,    
            "nu_true": self.nu_true,
            "errors": {
                "mu_l2": float(mu_err),
                "cov_fro": float(cov_err_frob),
                "Sigma_fro": float(sigma_err_frob),
            }
        }


        print(f"n={self.n}, p={self.p}, nu_true={self.nu_true:.3f} → nu_hat={nu_hat:.3f}")
        print(f"‖mu_hat - mu_true‖₂ = {mu_err:.4e}")
        print(f"‖Cov_hat - Cov_true‖_F = {cov_err_frob:.4e}")
        print(f"‖Sigma_hat - Sigma_true‖_F = {sigma_err_frob:.4e}")

        return results



if __name__ == "__main__":
    d = 5
# %%
    t = test_mvt_mm(n=12*90, p=5, nu_true=20, seed=18)

    _ = t.run()
    X = t._sample_mvt()
    cols = [f"x{i+1}" for i in range(d)]
    df = pd.DataFrame(X, columns=cols)

    
    K = np.array([16] * d, dtype=int)

    # est = MVTCFEstimator(
    #     X=df[cols].to_numpy(),
    #     K=K,
    #     decay=0.03,          # tweak if high-freqs dominate
    #     use_all_signs=False, # set True if you want all 2^(d-1) sign patterns
    #     dtype_ecf=np.complex64,
    #     L_box=7.0
    # )

    # out = est.fit(nu0=8.0, bounds=(2.05, 200.0))

    # # --- Report ---
    # print("\n=== True vs Estimated parameters ===")
    # print(f"Estimated ν:   {out['nu']:.4f}")
    # print(f"Optimize ok?:  {out['success']}, message: {out['message']}, iters: {out['nit']}")
    # print(f"Objective @ ν*: {out['fun']:.6e}")