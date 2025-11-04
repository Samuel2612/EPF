import math, itertools, time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List
from itertools import product
from scipy import optimize, special, stats

from ecf_einsum import ecf_grid_einsum




def simulate_mvt(n: int, mu: np.ndarray, Sigma_cov: np.ndarray, nu: float, rng=None) -> np.ndarray:
    """
    Simulate from a d-dim multivariate Student-t with mean mu, covariance Sigma_cov, dof nu (>2).
    Uses the scale-mixture representation: x = mu + L z / sqrt(u/nu), with z~N(0,I), u~chi2_nu.
    """
    if rng is None: rng = np.random.default_rng(7)
    d = mu.size
    Sigma_scale = ((nu - 2.0) / nu) * Sigma_cov  # convert covariance -> scale
    L = np.linalg.cholesky(Sigma_scale)
    z = rng.standard_normal((n, d))
    u = rng.chisquare(df=nu, size=n)            # chi^2_nu
    w = np.sqrt(u / nu).reshape(-1, 1)
    return mu + (z @ L.T) / w

def all_sign_patterns(d: int) -> np.ndarray:
    """Return sign matrix of shape (2^(d-1), d) with first sign fixed to +1 when requested."""

    rest = list(product([-1, 1], repeat=d-1))
    return np.array([[1] + list(r) for r in rest], dtype=int)


def build_axis_frequencies(a: np.ndarray, b: np.ndarray, s: np.ndarray, K: np.ndarray) -> List[np.ndarray]:
    """Return [t1,...,td] where tj[k] = pi * s_j * k / (b_j - a_j), k = 0..Kj-1."""
    alpha = np.pi * s / (b - a)
    return [alpha[j] * np.arange(K[j], dtype=float) for j in range(len(K))]

def student_t_cf_grid_slice(k1: int,
                            t_axes: List[np.ndarray],     # [t1,...,td]
                            mu: np.ndarray,               # (d,)
                            Sigma_scale: np.ndarray,      # (d,d) scale (not covariance!)
                            nu: float) -> np.ndarray:
    """
    Multivariate Student-t CF on the (K2 x ... x Kd) slice with fixed t1 = t_axes[0][k1].
    φ(t) = exp(i t·mu) * 2^{1-ν/2}/Γ(ν/2) * (√(ν t^T Σ t))^{ν/2} * K_{ν/2}(√(ν t^T Σ t))
    """
    d = len(mu)
    assert Sigma_scale.shape == (d, d)
    # Build broadcastable axes
    Ts = []
    for j, tj in enumerate(t_axes):
        shape = [1]*(d-1)
        shape[j-1 if j>0 else 0] = len(tj) if j>0 else 1
        # explicit shapes for d=5 (fast path)
    if d == 5:
        t1,t2,t3,t4,t5 = t_axes
        T1 = float(t1[k1])
        T2 = t2[:,None,None,None]
        T3 = t3[None,:,None,None]
        T4 = t4[None,None,:,None]
        T5 = t5[None,None,None,:]
        S  = Sigma_scale
        q  = (S[0,0]*T1**2 + S[1,1]*T2**2 + S[2,2]*T3**2 + S[3,3]*T4**2 + S[4,4]*T5**2)
        q += 2.0*( S[0,1]*T1*T2 + S[0,2]*T1*T3 + S[0,3]*T1*T4 + S[0,4]*T1*T5
                 + S[1,2]*T2*T3 + S[1,3]*T2*T4 + S[1,4]*T2*T5
                 + S[2,3]*T3*T4 + S[2,4]*T3*T5
                 + S[3,4]*T4*T5 )
        lin = mu[0]*T1 + mu[1]*T2 + mu[2]*T3 + mu[3]*T4 + mu[4]*T5
    else:
        raise NotImplementedError("This demo implements a fast slice path for d=5.")

    order = nu/2.0
    z = np.sqrt(np.maximum(0.0, nu) * np.maximum(0.0, q))
    g = np.ones_like(z, dtype=float)
    mask = (z > 0)
    if np.any(mask):
        pref = 2.0**(1.0 - order) / special.gamma(order)
        K = special.kv(order, z[mask])
        g[mask] = pref * (z[mask]**order) * K
        bad = ~np.isfinite(g[mask])
        if np.any(bad):
            zb = z[mask][bad]
            Kb = special.kv(order, zb)
            g[mask][bad] = pref * np.exp(order*np.log(zb)) * Kb
    return np.exp(1j*lin) * g  # (K2,...,K5)

def phase_slice(k1: int, t_axes: List[np.ndarray], a: np.ndarray) -> np.ndarray:
    """Compute exp(-i t·a) on the (K2 x ... x Kd) slice."""
    t1,t2,t3,t4,t5 = t_axes
    T1 = float(t1[k1])
    T2 = t2[:,None,None,None]
    T3 = t3[None,:,None,None]
    T4 = t4[None,None,:,None]
    T5 = t5[None,None,None,:]
    return np.exp(-1j*(T1*a[0] + T2*a[1] + T3*a[2] + T4*a[3] + T5*a[4]))


@dataclass
class MVTCFEstimator:
    X: np.ndarray                    # (N,d)
    K: Iterable[int]
    a: np.ndarray
    b: np.ndarray
    decay: float = 0.03
    use_all_signs: bool = False      # True -> 2^(d-1) signs
    dtype_ecf: np.dtype = np.complex64

    def __post_init__(self):
        self.N, self.d = self.X.shape
        self.K = np.asarray(self.K, dtype=int)
        assert len(self.K) == self.d

        # sample stats
        self.mu_hat = self.X.mean(axis=0)
        Xc = self.X - self.mu_hat
        self.Sigma_hat = (Xc.T @ Xc) / (self.N - 1)

        self.S = all_sign_patterns(self.d)
        if not self.use_all_signs:
            self.S = self.S[:1]  # only [+1,...,+1]
        self.t_axes_per_s = [build_axis_frequencies(self.a, self.b, s, self.K) for s in self.S]
        self._ecf_per_s = [self._ecf_single_sign(s) for s in self.S]

    def _Sigma_scale(self, nu: float) -> np.ndarray:
        return ((nu - 2.0)/nu) * self.Sigma_hat

    def _ecf_single_sign(self, s: np.ndarray) -> np.ndarray:
       
        return ecf_grid_einsum(self.X, self.a, self.b, s, self.K, dtype=self.dtype_ecf)
      

    def _loss(self, nu: float) -> float:
        Sig = self._Sigma_scale(nu)
        loss = 0.0
        for ecf, t_axes in zip(self._ecf_per_s, self.t_axes_per_s):
            for k1 in range(self.K[0]):
                phi = student_t_cf_grid_slice(k1, t_axes, self.mu_hat, Sig, nu)
                # exponential decay by frequency
                t1,t2,t3,t4,t5 = t_axes
                T1 = float(t1[k1])
                norms = np.sqrt(T1**2 + t2[:,None,None,None]**2 + t3[None,:,None,None]**2 +
                                t4[None,None,:,None]**2 + t5[None,None,None,:]**2)
                w = np.exp(-self.decay * norms)
                if k1 == 0:
                    w = w.copy()
                    w[0,0,0,0] = 0.0  # remove DC
                diff2 = np.abs(phi - ecf[k1])**2
                loss += float(np.sum(w*diff2))
        return loss / len(self.S)

    def _grad(self, nu: float, eps: float = 1e-4) -> np.ndarray:
        return np.array([(self._loss(nu+eps) - self._loss(nu-eps)) / (2.0*eps)])

    def fit(self, nu0: float = 8.0, bounds: Tuple[float,float]=(2.05, 200.0)):
        def fun(v): return self._loss(v[0])
        def jac(v): return self._grad(v[0])
        res = optimize.minimize(fun, x0=np.array([nu0]), jac=jac,
                                method="L-BFGS-B", bounds=[bounds])
        self.nu_hat = float(res.x[0])
        self.opt_result_ = res
        return {"mu": self.mu_hat, "Sigma_cov": self.Sigma_hat, "nu": self.nu_hat,
                "success": res.success, "message": res.message, "fun": res.fun}



def index_grid(Ks):
    grid = np.indices(Ks, dtype=np.float64)
    return np.moveaxis(grid, 0, -1)

def cos_basis(x, a, b, N):
    k = np.arange(N)
    v = np.cos(k * math.pi * (x - a) / (b - a))
    v[0] *= 0.5
    return v

def closed_form_moment_power(a, b, N, p):
    """COS weights J_k^{(p)} for y^p with basis convention 'k=0 half'."""
    L = b - a
    J = np.zeros(N, dtype=np.float64)
    J[0] = 0.5 * (b**(p + 1) - a**(p + 1)) / (p + 1)
    for k in range(1, N):
        il = 1j * k * math.pi
        pow_series = [(il)**m / math.factorial(m) for m in range(p + 1)]
        acc = 0.0
        for j in range(p + 1):
            S_j = sum(pow_series[:j + 1])
            I_j = math.factorial(j) / il**(j + 1) * (1 - np.exp(il) * S_j)
            acc += math.comb(p, j) * a**(p - j) * L**(j + 1) * I_j.real
        J[k] = acc
    return J

def build_row_core(G0_mat, weights):
    return (weights[:, None] * G0_mat).sum(0, keepdims=True)

def feature_contraction(x_feat, cores, a, b):
    vec = None
    for i, G in enumerate(cores[1:], start=1):
        N = G.shape[1]
        basis = cos_basis(x_feat[i-1], a[i], b[i], N)
        M = np.tensordot(G, basis, axes=([1],[0]))
        vec = M if vec is None else vec @ M
    return vec  # (r1,1)

def tt_svd_robust(tensor, eps):
    """Robust TT-SVD (rank>=1) – see your cond_moments_tt.py for details."""
    T = tensor.copy()
    dims = T.shape
    d = len(dims)
    norm2 = np.linalg.norm(T)**2
    thr = (eps / math.sqrt(max(d-1,1)))**2 * norm2
    cores = []
    ranks = [1]
    unfold = T
    for k in range(d-1):
        m = ranks[-1] * dims[k]
        unfold = unfold.reshape(m, -1)
        U,S,Vh = np.linalg.svd(unfold, full_matrices=False)
        tail = np.cumsum(S[::-1]**2)[::-1]
        # keep all if none below thr; always keep at least 1
        mask = tail <= thr
        r = int(np.argmax(mask)) + 1 if mask.any() else len(S)
        r = max(1, r)
        cores.append(U[:, :r].reshape(ranks[-1], dims[k], r))
        ranks.append(r)
        unfold = (S[:r, None] * Vh[:r])
    cores.append(unfold.reshape(ranks[-1], dims[-1], 1))
    ranks.append(1)
    return cores, ranks

class MultiCOS:
    """
    COS/TT class for multivariate Student-t.
    Dims: first dim is Y (the one we take conditional moments of), others are features X.
    """

    def __init__(self, Ks: Iterable[int], a: np.ndarray, b: np.ndarray,
                 mu: np.ndarray, Sigma_cov: np.ndarray, nu: float,
                 use_all_signs: bool = False):
        self.Ks = np.asarray(Ks, int)
        self.a  = np.asarray(a, float)
        self.b  = np.asarray(b, float)
        self.mu = np.asarray(mu, float)
        self.nu = float(nu)
        self.d  = len(self.Ks)
        assert self.mu.shape == (self.d,)
        assert Sigma_cov.shape == (self.d, self.d)
        # convert covariance -> scale for Student-t CF
        self.Sigma_scale = ((self.nu - 2.0) / self.nu) * Sigma_cov
        self.S = all_sign_patterns(self.d)
        if not use_all_signs:
            self.S = self.S[:1]  # single sign pattern by default
        self._cores = None
        self._row_cores = None

    @staticmethod
    def _chf_MVT(t: np.ndarray, mu: np.ndarray, Sigma_scale: np.ndarray, nu: float) -> np.ndarray:
        """
        Evaluate φ(t) for MVT_d(nu, mu, Sigma_scale).
        t: (..., d) array; returns shape t.shape[:-1].
        """
        tf = t.reshape(-1, t.shape[-1])
        lin = tf @ mu                  # (N,)
        q   = np.einsum("ni,ij,nj->n", tf, Sigma_scale, tf, optimize=True)
        z   = np.sqrt(np.maximum(0.0, nu) * np.maximum(0.0, q))
        order = nu/2.0
        g = np.ones_like(z, dtype=float)
        mask = (z > 0)
        if np.any(mask):
            pref = 2.0**(1.0 - order) / special.gamma(order)
            K = special.kv(order, z[mask])
            g[mask] = pref * (z[mask]**order) * K
            bad = ~np.isfinite(g[mask])
            if np.any(bad):
                zb = z[mask][bad]
                Kb = special.kv(order, zb)
                g[mask][bad] = pref * np.exp(order*np.log(zb)) * Kb
        return np.exp(1j * lin) * g.reshape(tf.shape[0])  


    def build_A_tensor(self) -> np.ndarray:
        """
        Compute A_{k1..kd} = 2 * prod_i (1/(b_i-a_i)) * sum_s Re[exp(-i t·a) φ(t)],
        with t = π s ⊙ k / (b-a), on the full K grid.
        Efficient slice construction to keep memory low.
        """
        K = self.Ks
        d = self.d
        L = self.b - self.a
        factor = 2.0 / L.prod()
        A = np.zeros(tuple(K.tolist()), dtype=np.float64)

        for s in self.S:
            # frequency axes for this sign pattern
            t_axes = build_axis_frequencies(self.a, self.b, s, K)

            # process slices over k1
            for k1 in range(K[0]):
                phi = student_t_cf_grid_slice(k1, t_axes, self.mu, self.Sigma_scale, self.nu)
                phs = phase_slice(k1, t_axes, self.a)
                A[k1] += np.real(phs * phi)

        A *= factor
        return A

    def tt_from_A(self, A: np.ndarray, eps: float = 1e-12):
        cores, ranks = tt_svd_robust(A, eps=eps)
        self._cores = cores
        self._ranks = ranks
        return cores, ranks

    def prepare_row_cores(self, p_max: int = 4):
        if self._cores is None:
            raise RuntimeError("Call tt_from_A() first.")
        G0 = self._cores[0].reshape(self.Ks[0], -1)
        rows = {}
        for p in range(p_max + 1):
            w = closed_form_moment_power(self.a[0], self.b[0], self.Ks[0], p)
            rows[p] = build_row_core(G0, w)
        self._row_cores = rows
        self._tilde_G0  = rows[0]

    def cond_raw_moments(self, x_feat: np.ndarray, p_list=(1,2,3,4)) -> dict:
        if self._row_cores is None:
            raise RuntimeError("Call prepare_row_cores() first.")
        P = feature_contraction(x_feat, self._cores, self.a, self.b)  # (r1,1)
        norm = float((self._tilde_G0 @ P).squeeze())
        out = {}
        for p in p_list:
            num = float((self._row_cores[p] @ P).squeeze())
            out[p] = num / norm
        return out



def analytic_conditional_student_t_moments(x_feat: np.ndarray,
                                           mu: np.ndarray,
                                           Sigma_cov: np.ndarray,
                                           nu: float) -> Tuple[float,float,float,float]:
    """
    Partition (Y,X) with Y scalar (dim 0), X in R^{d-1}.
    Joint is MVT_d(nu, mu, Sigma_scale) where Sigma_cov is the covariance.
    Conditional Y|X=x is Student-t with:
      nu' = nu + d_x
      mu_c = mu_y + Σ_yx Σ_xx^{-1} (x - mu_x)
      scale_c = (Σ_yy - Σ_yx Σ_xx^{-1} Σ_xy) * (nu + δ) / (nu + d_x),
        δ = (x - mu_x)^T Σ_xx^{-1} (x - mu_x)
    Moments (require nu' > 4):
      Var = nu'/(nu'-2) * scale_c
      E[Y]   = mu_c
      E[Y^2] = Var + mu_c^2
      E[Y^3] = mu_c^3 + 3*mu_c*Var
      E[Y^4] = mu_c^4 + 6*mu_c^2*Var + 3 * nu'^2/( (nu'-2)*(nu'-4) ) * scale_c**2
    """
    d = len(mu)
    d_x = d - 1
    mu_y, mu_x = mu[0], mu[1:]
    Σ = Sigma_cov
    Σ_xx = Σ[1:,1:]
    Σ_yx = Σ[0,1:]
    Σ_xy = Σ_yx[:,None]
    Σ_xx_inv = np.linalg.inv(Σ_xx)

    diff = (x_feat - mu_x)
    mu_c = float(mu_y + Σ_yx @ Σ_xx_inv @ diff)
    base_scale = float(Σ[0,0] - Σ_yx @ Σ_xx_inv @ Σ_xy)  # covariance version, will be turned to scale below

    # We need the *scale* of the conditional t, not covariance.
    # For the *joint*, covariance = nu/(nu-2) * scale  => scale = (nu-2)/nu * covariance.
    Σ_scale_joint = ((nu - 2.0) / nu) * Σ
    Σs_xx = Σ_scale_joint[1:,1:]
    Σs_yx = Σ_scale_joint[0,1:]
    Σs_xy = Σs_yx[:,None]
    base_scale_scale = float(Σ_scale_joint[0,0] - Σs_yx @ np.linalg.inv(Σs_xx) @ Σs_xy)

    δ = float(diff.T @ Σ_xx_inv @ diff)
    nu_c = nu + d_x
    scale_c = base_scale_scale * (nu + δ) / (nu + d_x)  # Student-t *scale* parameter

    if nu_c <= 4:
        raise ValueError("nu' ≤ 4; the 4th moment does not exist.")

    Var = nu_c/(nu_c - 2.0) * scale_c
    m1  = mu_c
    m2  = Var + mu_c**2
    m3  = mu_c**3 + 3*mu_c*Var
    m4  = mu_c**4 + 6*mu_c**2*Var + 3 * (nu_c**2)/((nu_c-2)*(nu_c-4)) * (scale_c**2)
    return m1, m2, m3, m4


if __name__ == "__main__":
    rng = np.random.default_rng(123)

    # --- True parameters for synthetic 5D MVT ---
    d = 5
    mu_true = np.array([0.02, -0.01, 0.015, 0.0, -0.005])
    sds = np.array([0.08, 0.06, 0.07, 0.09, 0.05])
    corr = 0.3
    R = (1 - corr) * np.eye(d) + corr * np.ones((d, d))
    Sigma_cov_true = (sds[:, None] @ sds[None, :]) * R
    nu_true = 10.0  # ensure nu'+ > 4

    # # --- (1) simulate N = 12*365 samples ---
    # N = 12 * 365
    # X = simulate_mvt(N, mu_true, Sigma_cov_true, nu_true, rng=rng)
    # df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(d)])

    # # --- COS box [a,b] from sample stats (mu ± 7σ) ---
    # mu_samp = df.to_numpy().mean(axis=0)
    # Sigma_samp = np.cov(df.to_numpy().T, bias=False)
    # std = np.sqrt(np.clip(np.diag(Sigma_samp), 1e-12, np.inf))
    # a = mu_samp - 7*std
    # b = mu_samp + 7*std

    # # --- estimate nu via ECF-vs-CF matching on the COS grid ---
    # K = np.array([64]*d, int)
    # est = MVTCFEstimator(df.to_numpy(), K=K, a=a, b=b, decay=0.03, use_all_signs=False)
    # fit_out = est.fit(nu0=8.0, bounds=(2.05, 200.0))
    # nu_hat = fit_out["nu"]
    # print("\n=== ECF-vs-CF Fit ===")
    # print(f"True nu = {nu_true:.4f} | Estimated nu = {nu_hat:.4f} | success={fit_out['success']} | fun={fit_out['fun']:.3e}")
    # mu_hat, Sigma_cov_hat = fit_out["mu"], fit_out["Sigma_cov"]

    # --- build COS tensor A_k using the MVT CF, and TT compress ---
    
    K = np.array([64]*d, int)
    mu_hat = mu_true
    Sigma_cov_hat = Sigma_cov_true
    std = np.sqrt(np.clip(np.diag(Sigma_cov_hat), 1e-12, np.inf))
    a = mu_hat - 7*std
    b = mu_hat + 7*std
    nu_hat = 10
    print("\nBuilding A_k tensor from the Student-t CF ...")
    cos = MultiCOS(Ks=K, a=a, b=b, mu=mu_hat, Sigma_cov=Sigma_cov_hat, nu=nu_hat, use_all_signs=True)
    t0 = time.time()
    A = cos.build_A_tensor()
    print(f"A built: shape={A.shape}, time={time.time()-t0:.2f}s")

    print("TT-SVD compression...")
    t1 = time.time()
    cores, ranks = cos.tt_from_A(A, eps=1e-12)
    print(f"TT ranks: {ranks} | time={time.time()-t1:.2f}s")

    cos.prepare_row_cores(p_max=4)

    # --- Evaluate conditional moments at random feature points ---
    rng_eval = np.random.default_rng(321)
    xs = rng_eval.normal(mu_hat[1:], np.sqrt(np.diag(Sigma_cov_hat))[1:], size=(50, d-1))

    errs = {p: [] for p in (1,2,3,4)}
    for x_feat in xs:
        m_tt = cos.cond_raw_moments(x_feat, p_list=(1,2,3,4))
        m_an = analytic_conditional_student_t_moments(x_feat, mu_hat, Sigma_cov_hat, nu_hat)
        for j,p in enumerate((1,2,3,4)):
            errs[p].append(abs(m_tt[p] - m_an[j]))

    print("\n=== Conditional raw-moment errors (TT vs analytic Student-t; using fitted params) ===")
    for p in (1,2,3,4):
        arr = np.array(errs[p])
        print(f" p={p}:  max={arr.max():.3e}  mean={arr.mean():.3e}")

    print("\nDone.")
