import numpy as np
from numpy.random import Generator, PCG64
from scipy.optimize import minimize
from scipy.stats import chi2
from scipy.special import kv, gammaln


class StudentTModel:
    """
    Multivariate Student-t with Samuelson scaling on the first component.

    Characteristic function (correct Bessel-K form):
      φ(u) = exp(i u·m) * C(ν) * δ(u)^{a} * K_a(z),
      where a = ν/2, δ(u) = sqrt( ũᵀ Σ ũ ), z = sqrt(ν) * δ(u),
            ũ is u with Samuelson scaling on the first axis,
            C(ν) = ν^{a} / (2^{a-1} Γ(a)).

    We provide analytic ∂φ/∂μ and ∂φ/∂Σ, and a robust numeric ∂φ/∂ν.

    Notes
    -----
    • Samuelson scaling only affects the FIRST component:
        ũ₁ = u₁ * τ^{ -α/2 }, ũ₂..d = u₂..d
    • Derivatives are returned per frequency row (J, ...) matching u.
    • ∂φ/∂Σ is a (J, d, d) array; symmetric in the last two dims.
    """

    def __init__(self, d, k, mu=None, B=None, Sigma=None, alpha=0.6, nu=10.0):
        self.d = int(d)
        self.k = int(k)
        self.mu = np.zeros(self.d) if mu is None else np.asarray(mu, float).reshape(self.d)
        self.B = np.zeros((self.d, self.k)) if B is None else np.asarray(B, float).reshape(self.d, self.k)
        self.Sigma = np.eye(self.d) if Sigma is None else np.asarray(Sigma, float).reshape(self.d, self.d)
        self.alpha = float(alpha)
        self.nu = float(nu)

    # ----------- initialisation heuristics -----------
    def init_from_data(self, X, tau=None):
        X = np.asarray(X, float)
        n, d = X.shape
        self.mu = np.nanmedian(X, axis=0)

        S = np.cov(X.T)
        lam = 1e-3 * np.trace(S) / d if np.isfinite(S).all() else 1e-3
        self.Sigma = S + lam * np.eye(d)

        self.alpha = 0.6
        if tau is not None:
            t = np.asarray(tau, float).ravel()
            r = X[:, 0].ravel()
            t = np.clip(t, 1e-9, np.inf)
            lt = np.log(t)
            qs = np.quantile(lt, np.linspace(0, 1, 6))
            centres, vbin = [], []
            for lo, hi in zip(qs[:-1], qs[1:]):
                sel = (lt >= lo) & (lt < hi)
                if sel.any():
                    centres.append(np.exp((lo + hi) / 2))
                    vbin.append(np.var(r[sel]))
            if len(vbin) >= 2 and np.all(np.array(vbin) > 0):
                slope, _ = np.polyfit(np.log(centres), np.log(vbin), 1)
                self.alpha = float(np.clip(-slope, 0.05, 3.0))

        r = X[:, 0]
        m2 = np.mean((r - r.mean()) ** 2)
        m4 = np.mean((r - r.mean()) ** 4)
        nu = 2.0 + (m2 ** 2) / np.clip(m4 - m2 ** 2, 1e-12, np.inf)
        self.nu = float(np.clip(nu, 2.1, 500.0))
        self.B = np.zeros_like(self.B)

    # ----------- helpers for CF & derivatives -----------
    def _scale_first_axis_u(self, u, tau):
        u = np.asarray(u, float)
        Du = u.copy()
        if np.ndim(tau) == 0:
            s = float(tau) ** (-self.alpha / 2.0)
            Du[:, 0] = Du[:, 0] * s
        else:
            t = np.asarray(tau, float).reshape(u.shape[0], 1)
            s = t ** (-self.alpha / 2.0)
            Du[:, 0] = Du[:, 0] * s[:, 0]
        return Du

    def _C_nu(self, nu):
        # C(ν) = ν^{ν/2} / (2^{ν/2 - 1} Γ(ν/2))
        a = 0.5 * nu
        logC = a * np.log(nu) - (a - 1.0) * np.log(2.0) - gammaln(a)
        return np.exp(logC)

    def _delta_and_bessel(self, u, tau, Z=None):
        """
        Compute φ(u), plus intermediate pieces needed for derivatives.
        Returns dict with:
          phi, exp_ium, Cnu, a, z, delta, u_tilde, Ka, Ka_m, Ka_p
        """
        u = np.asarray(u, float)
        J, d = u.shape

        # mean with optional drivers
        if Z is None:
            m = self.mu.reshape(1, d)
        else:
            Z = np.asarray(Z, float)
            if Z.ndim == 1:
                Z = np.tile(Z.reshape(1, -1), (J, 1))
            m = self.mu.reshape(1, d) + Z @ self.B.T

        exp_ium = np.exp(1j * np.einsum("jd,jd->j", u, m))

        # Samuelson scaling on u -> ũ
        ut = self._scale_first_axis_u(u, tau)

        # δ(u) = sqrt(utᵀ Σ ut)
        q = np.einsum("jd,dd,jd->j", ut, self.Sigma, ut)
        delta = np.sqrt(np.maximum(q, 0.0))

        nu = self.nu
        a = 0.5 * nu
        z = np.sqrt(nu) * delta
        Cnu = self._C_nu(nu)

        # Handle small z via asymptotics to avoid blow-ups:
        # K_a(z) ~ 2^{a-1} Γ(a) z^{-a}  as z -> 0+
        # => δ^a K_a(√ν δ) -> 2^{a-1} Γ(a) ν^{-a/2}
        Ka = np.empty(J, float)
        Ka_m = np.empty(J, float)  # K_{a-1}(z)
        Ka_p = np.empty(J, float)  # K_{a+1}(z)

        tiny = 1e-9
        small = z < tiny
        if np.any(small):
            const = (2.0 ** (a - 1.0)) * np.exp(gammaln(a)) * (z[small] ** (-a))
            Ka[small] = const
            # For derivatives, we still need K_{a±1}. Use leading-order:
            Ka_m[small] = (2.0 ** (a - 2.0)) * np.exp(gammaln(a - 1.0)) * (z[small] ** (-(a - 1.0)))
            Ka_p[small] = (2.0 ** (a)) * np.exp(gammaln(a + 1.0)) * (z[small] ** (-(a + 1.0)))

        if np.any(~small):
            zc = z[~small]
            Ka[~small] = kv(a, zc)
            Ka_m[~small] = kv(a - 1.0, zc)
            Ka_p[~small] = kv(a + 1.0, zc)

        # φ(u)
        g = np.empty(J, complex)
        g[:] = (delta ** a) * Ka
        phi = exp_ium * (Cnu * g)

        return {
            "phi": phi,
            "exp_ium": exp_ium,
            "Cnu": Cnu,
            "a": a,
            "z": z,
            "delta": delta,
            "u_tilde": ut,
            "Ka": Ka,
            "Ka_m": Ka_m,
            "Ka_p": Ka_p,
        }

    # ----------- characteristic function -----------
    def phi(self, u, tau, Z=None):
        stuff = self._delta_and_bessel(u, tau, Z)
        return stuff["phi"]

    # ----------- partial derivatives of the CF -----------
    def dphi_dmu(self, u, tau, Z=None):
        """
        ∂φ/∂μ  (shape: J × d)
        Since φ(u) = e^{i u·m} * G(u), we have ∂φ/∂μ = i u φ(u).
        """
        u = np.asarray(u, float)
        phi = self.phi(u, tau, Z)
        return (1j * u) * phi.reshape(-1, 1)

    def dphi_dSigma(self, u, tau, Z=None):
        """
        ∂φ/∂Σ  (shape: J × d × d)

        Using:
          φ(u) = e^{i u·m} C(ν) δ^a K_a(z),  a=ν/2, z=√ν δ, δ = sqrt(utᵀ Σ ut)
          dδ/dΣ = (1 / (2 δ)) * (ut utᵀ)

          d/dδ [δ^a K_a(z)]  with z=√ν δ:
            g'(δ) = a δ^{a-1} K_a(z) - 0.5 √ν δ^{a} [ K_{a-1}(z) + K_{a+1}(z) ]
        Hence:
          ∂φ/∂Σ = e^{i u·m} C(ν) g'(δ) * (1/(2 δ)) * (ut utᵀ)
        """
        pieces = self._delta_and_bessel(u, tau, Z)
        exp_ium = pieces["exp_ium"]
        Cnu = pieces["Cnu"]
        a = pieces["a"]
        z = pieces["z"]
        delta = pieces["delta"]
        ut = pieces["u_tilde"]
        Ka = pieces["Ka"]
        Ka_m = pieces["Ka_m"]
        Ka_p = pieces["Ka_p"]

        J, d = ut.shape
        out = np.zeros((J, d, d), dtype=complex)

        # handle δ=0 safely: derivative is 0 at u=0 exactly
        mask_pos = delta > 0.0
        if np.any(mask_pos):
            dlt = delta[mask_pos]
            zlt = z[mask_pos]
            gprime = (a * (dlt ** (a - 1.0)) * Ka[mask_pos]
                      - 0.5 * np.sqrt(self.nu) * (dlt ** a) * (Ka_m[mask_pos] + Ka_p[mask_pos]))
            pref = (exp_ium[mask_pos] * Cnu * gprime) / (2.0 * dlt)  # (m,)

            v = ut[mask_pos]  # (m,d)
            # (m,d,d) = pref[:,None,None] * v[:,:,None] * v[:,None,:]
            out[mask_pos] = (pref[:, None, None] * (v[:, :, None] * v[:, None, :])).astype(complex)

        return out

    def dphi_dnu(self, u, tau, Z=None, h=1e-5):
        """
        ∂φ/∂ν via symmetric finite differences on ν.
        This includes all ν-dependence (C(ν), order a=ν/2, and z=√ν δ).
        """
        nu0 = self.nu
        self.nu = nu0 + h
        ph_plus = self.phi(u, tau, Z)
        self.nu = nu0 - h
        ph_minus = self.phi(u, tau, Z)
        self.nu = nu0
        return (ph_plus - ph_minus) / (2.0 * h)

    # ----------- sampling -----------
    def sample(self, n, tau, Z=None, rng=None):
        """
        Draw n samples from the model at lead time tau (scalar).
        Uses the Gaussian scale-mixture representation of Student-t.
        Applies Samuelson scaling on the FIRST component.
        """
        rng = rng or Generator(PCG64())
        n = int(n)

        s = rng.chisquare(self.nu, size=n)
        eps = rng.standard_normal((n, self.d))
        L = np.linalg.cholesky(self.Sigma)
        base = (np.sqrt(self.nu / s)[:, None]) * (eps @ L.T)

        if Z is None:
            mean = self.mu
        else:
            z = np.asarray(Z, float)
            if z.ndim > 1:
                z = z[0]
            mean = self.mu + self.B @ z

        D = np.ones(self.d)
        D[0] = float(tau) ** (-self.alpha / 2.0)
        return mean.reshape(1, self.d) + base * D