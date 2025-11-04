# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:56:33 2025

@author: samue
"""

import numpy as np
import scipy.stats as st
import time

def safe_div(num, den, eps=1e-12, big=1e12):

    num = np.asarray(num)
    den = np.asarray(den)

    out = np.empty_like(np.broadcast_to(num, np.broadcast(num, den).shape))

    mask = np.abs(den) >= eps
    out[mask]  = num[mask] / den[mask]
    out[~mask] = np.sign(num[~mask]) * big       #t

    return out

def safe_inv(x, eps=1e-12, big=1e12):
    """
    Safe reciprocal.

    • If |x| ≥ eps  → 1 / x
    • If |x| < eps  → ±big   (sign follows x; 0 gets +big)

    Works on scalars or NumPy arrays (broadcasts automatically).
    """
    x   = np.asarray(x, dtype=float)
    out = np.empty_like(x)

    mask = np.abs(x) >= eps
    out[mask]  = 1.0 / x[mask]
    out[~mask] = np.sign(x[~mask]) * big    # sign(0) is 0 → +big

    return out

class link_softplus:
    """
    Soft-plus link:
        inverse (η → μ) : μ = log(1 + exp(η))           # always > 0
        link    (μ → η) : η = log(exp(μ) - 1),  μ > 0   # inverse of soft-plus
    """

    def __init__(self):
        pass

   
    def link(self, x: np.ndarray) -> np.ndarray:
        """
        η = g(μ) with μ > 0.  Uses expm1 for numerical stability.
        """
        x_clipped = np.clip(x, 1e-20, None)                 # avoid log(0)
        return np.log(np.expm1(x_clipped))                  # log(exp(x) - 1)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        """
        μ = g^{-1}(η) = softplus(η).  For large η clip to avoid overflow.
        """
        return np.log1p(np.exp(np.clip(x, None, 40.0)))     # log(1 + exp(x))

    
    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        dμ/dη = sigmoid(η) = 1 / (1 + exp(-η))
        """
        z = np.clip(x, None, 40.0)
        return 1.0 / (1.0 + np.exp(-z))

    def inverse_second_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        d²μ/dη² = sigmoid(η) * (1 - sigmoid(η))
        """
        s = self.inverse_derivative(x)
        return s * (1.0 - s)

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        dη/dμ = exp(μ) / (exp(μ) - 1) = 1 / (1 - exp(-μ))
        """
        x_clipped = np.clip(x, 1e-12, None)                 # keep μ > 0
        return 1.0 / (1.0 - np.exp(-x_clipped))



class link_log():
    """
    The log-link function.
    """

    def __init__(self):
        pass

    def link(self, x: np.ndarray) -> np.ndarray:
        return np.log(np.clip(x, 1e-20, None))

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.exp(np.clip(x, None, 1e2))

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.exp(np.clip(x, None, 1e2))
    
    def inverse_second_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.exp(np.clip(x, None, 1e2))

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / x
    
    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        return -1 / np.clip(x**2, 1e-300, None)






def robust_log(x, eps = 1e-20):
    """Safe log that never sees 0 or negatives."""
    return np.log(np.clip(x, eps, None))


def robust_exp(x, clip_high = 80.0):
    """Safe exp that never overflows float64 (`exp(80) ≈ 5.5 e34`)."""
    return np.exp(np.clip(x, None, clip_high))


# ------------------------------------------------------------------
# Log–Identity link (Narajewski & Ziel, 2020)
# ------------------------------------------------------------------
class link_logident():



    def link(self, x):
        return np.where(x <= 1.0, robust_log(x), x - 1.0)

    
    def inverse(self, x):
        return np.where(x <= 0.0, robust_exp(x), x + 1.0)


    def inverse_derivative(self, x):
        return np.where(x <= 0.0, robust_exp(x), 1.0)

    def inverse_second_derivative(self, x):
        return np.where(x <= 0.0, robust_exp(x), 0.0)

    def link_derivative(self, x):
        return np.where(x <= 1.0, 1.0 / np.clip(x, 1e-20, None), 1.0)

    def link_second_derivative(self, x):
        return np.where(x <= 1.0, -1.0 / np.clip(x, 1e-20, None) ** 2, 0.0)



class link_id():
    """
    The identity link function.

    """

    def __init__(self):
        pass

    def link(self, x):
        return x

    def inverse(self, x):
        return x

    def inverse_derivative(self, x):
        return np.ones_like(x)

    def link_derivative(self, x):
        return np.ones_like(x)
    
    def inverse_second_derivative(self, x):
        return np.zeros_like(x)
    
    def link_second_derivative(self, x):
        return np.zeros_like(x)



def _c_omega_w(nu, tau):
    """
    Helper used by both conversions.
    Returns c, omega, w exactly as defined in GAMLSS notes:

        omega = -ν / τ
        w     = exp(1/τ²)   (≈ 1  when τ is large)
        c     = [½ (w-1)(w cosh(2ω)+1)]^(-½)
    """
    tau   = np.asarray(tau, dtype=float)
    nu    = np.asarray(nu,  dtype=float)

    rtau  = 1.0 / tau
    w     = np.exp(rtau**2)
    omega = -nu * rtau
    c     = (0.5 * (w - 1.0) * (w * np.cosh(2.0 * omega) + 1.0)) ** (-0.5)
    return c, omega, w



def jsu_reparam_to_original(mu, sigma, nu, tau):
    """
    Convert from the re-parameterised GAMLSS form
    JSU(μ, σ, ν, τ)  to the *original* form  JSU₀(μ₁, σ₁, ν₁, τ₁).

    Parameters
    ----------
    mu, sigma, nu, tau : array_like or float
        Broadcastable arrays of equal shape.

    Returns
    -------
    mu1, sigma1, nu1, tau1 : ndarray
        Parameters of JSU₀.
    """
    mu     = np.asarray(mu,    dtype=float)
    sigma  = np.asarray(sigma, dtype=float)
    nu     = np.asarray(nu,    dtype=float)
    tau    = np.asarray(tau,   dtype=float)

    c, omega, w = _c_omega_w(nu, tau)

    mu1    = mu - c * sigma * np.sqrt(w) * np.sinh(omega)
    sigma1 = c * sigma
    nu1    = -nu
    tau1   = tau
    return mu1, sigma1, nu1, tau1



def jsu_original_to_reparam(mu1, sigma1, nu1, tau1):
    """
    Inverse transformation: JSU₀(μ₁, σ₁, ν₁, τ₁) → JSU(μ, σ, ν, τ).
    """
    mu1    = np.asarray(mu1,    dtype=float)
    sigma1 = np.asarray(sigma1, dtype=float)
    nu1    = np.asarray(nu1,    dtype=float)
    tau1   = np.asarray(tau1,   dtype=float)

    # reverse the simple parts first
    tau    = tau1
    nu     = -nu1

    c, omega, w = _c_omega_w(nu1, tau1)

    sigma  = sigma1 / c
    mu     = mu1 - sigma1 * np.sqrt(w) * np.sinh(nu1/tau1)
    return mu, sigma, nu, tau

class NO:
    """Corresponds to GAMLSS NO:  σ is the *standard deviation* (> 0)."""

    def __init__(self, loc_link=link_id(), scale_link=link_log()):
        self.n_of_p   = 2  # μ, σ (variance)
        self.loc_link = loc_link
        self.scale_link = scale_link
        self.links     = [self.loc_link, self.scale_link]



    def theta_to_params(self, theta):
        mu, sigma = theta.T
        sigma = np.clip(sigma, 1e-50, None)  # robustness
        return mu, sigma

    def link_function(self, x, p=0):            
        return self.links[p].link(x)
    def link_inverse(self,  x, p=0):            
        return self.links[p].inverse(x)
    def link_function_derivative(self, x, p=0): 
        return self.links[p].link_derivative(x)
    def link_inverse_derivative(self, x, p=0):  
        return self.links[p].inverse_derivative(x)
    def link_inverse_second_derivative(self, x, p=0):
        return self.links[p].inverse_second_derivative(x)
    def link_second_derivative(self, x, p=0):  
        return self.links[p].link_second_derivative(x)


    @staticmethod
    def _dmu(y, mu, sigma):
        return (y - mu) / sigma**2
    
    @staticmethod
    def _ddmu(y, mu, sigma):
        return  -(1 / sigma**2)

    @staticmethod
    def _dsigma(y, mu, sigma):
        return ((y - mu) ** 2 - sigma**2) / (sigma**3)

    @staticmethod
    def _ddsigma(y, mu, sigma):
        return -(2 / (sigma**2))



    def dll(self, y, theta, p):
        mu, sigma = self.theta_to_params(theta)
        if p == 0:
            return self._dmu(y, mu, sigma)
        elif p == 1:
            return self._dsigma(y, mu, sigma)
        else:
            raise ValueError("p must be 0 (mu) or 1 (variance).")
            
    
    def ddll(self, y, theta, p):
        mu, sigma = self.theta_to_params(theta)
        if p == 0:
            return self._ddmu(y, mu, sigma)
        elif p == 1:
            return self._ddsigma(y, mu, sigma)
        else:
            raise ValueError("p must be 0 (mu) or 1 (variance).")

    
    def pdf(self, y, theta):
        mu, sigma = self.theta_to_params(theta)
        return st.norm.pdf(y, loc=mu, scale=np.sqrt(sigma))

    def cdf(self, y, theta):
        mu, sigma = self.theta_to_params(theta)
        return st.norm.cdf(y, loc=mu, scale=np.sqrt(sigma))

    def init_val(self, y):
        mu_   = np.mean(y) 
        sigma_   = np.std(y)
        params = np.array([mu_, sigma_])
        out    = np.tile(params, (y.shape[0], 1))
        # out[:, 0] = (out[:, 0] + y)/2
        return out


class NO2:
    """Corresponds to GAMLSS NO2:  σ is the *variance* (> 0)."""

    def __init__(self, loc_link=link_id(), scale_link=link_log()):
        self.n_of_p   = 2  # μ, σ (variance)
        self.loc_link = loc_link
        self.scale_link = scale_link
        self.links     = [self.loc_link, self.scale_link]


    def theta_to_params(self, theta):
        mu, sigma = theta.T
        # sigma = np.clip(sigma, 1e-100, None)  # robustness
        return mu, sigma

    def link_function(self, x, p=0):            
        return self.links[p].link(x)
    def link_inverse(self,  x, p=0):            
        return self.links[p].inverse(x)
    def link_function_derivative(self, x, p=0): 
        return self.links[p].link_derivative(x)
    def link_inverse_derivative(self, x, p=0):  
        return self.links[p].inverse_derivative(x)
    def link_inverse_second_derivative(self, x, p=0):
        return self.links[p].inverse_second_derivative(x)
    def link_second_derivative(self, x, p=0):  
        return self.links[p].link_second_derivative(x)


    @staticmethod
    def _dmu(y, mu, sigma):
        return (y - mu) / sigma
    
    @staticmethod
    def _ddmu(y, mu, sigma):
        return -(1 / sigma)

    @staticmethod
    def _dsigma(y, mu, sigma):
        return 0.5 * ((y - mu) ** 2 - sigma) / (sigma ** 2)

    @staticmethod
    def _ddsigma(y, mu, sigma):
        return -(1 / (2 * sigma**2))



    def dll(self, y, theta, p):
        mu, sigma = self.theta_to_params(theta)
        if p == 0:
            return self._dmu(y, mu, sigma)
        elif p == 1:
            return self._dsigma(y, mu, sigma)
        else:
            raise ValueError("p must be 0 (mu) or 1 (variance).")
            
    
    def ddll(self, y, theta, p):
        mu, sigma = self.theta_to_params(theta)
        if p == 0:
            return np.clip(self._ddmu(y, mu, sigma), None, -1e-15)
        elif p == 1:
            return np.clip(self._ddsigma(y, mu, sigma), None, -1e-15)
        else:
            raise ValueError("p must be 0 (mu) or 1 (variance).")

    
    def pdf(self, y, theta):
        mu, sigma = self.theta_to_params(theta)
        return st.norm.pdf(y, loc=mu, scale=np.sqrt(sigma))

    def cdf(self, y, theta):
        mu, sigma = self.theta_to_params(theta)
        return st.norm.cdf(y, loc=mu, scale=np.sqrt(sigma))

    def init_val(self, y):
        mu_   = np.mean(y) 
        sigma_   = np.var(y)
        params = np.array([mu_, sigma_])
        out    = np.tile(params, (y.shape[0], 1))
        return out


class JSUo:
    """
    A Johnson SU (original) distribution class for GAMLSS-like usage.
    
    This implementation:
      - Defines score (first derivative) for each parameter
      - Defines exact second derivatives
      - Uses scipy.stats.johnsonsu for PDF and CDF
      - Allows link functions for each parameter
      - Provides initial values for iterative fitting
    """

    def __init__(
        self,
        loc_link=link_id(),
        scale_link=link_log(),
        skew_link=link_id(),
        kurt_link=link_log(),
    ):

        self.n_of_p = 4
        self.loc_link = loc_link
        self.scale_link = scale_link
        self.skew_link = skew_link
        self.kurt_link = kurt_link
        self.links = [
            self.loc_link,
            self.scale_link,
            self.skew_link,
            self.kurt_link,
        ]


    def theta_to_params(self, theta):
        """
        Extract mu, sigma, nu, tau from theta.


        """
        # Each column of theta is the linear predictor for one parameter
        mu, sigma, nu, tau = theta.T  # relies on theta.shape == (n,4)
        return mu, sigma, nu, tau
            
    def link_function(self, x, p=0):
        """
        Apply link function to raw parameter x.


        """
        return self.links[p].link(x)

    def link_inverse(self, x, p=0):
        """
        Apply inverse link function to linear predictor x.


        """
        return self.links[p].inverse(x)

    def link_function_derivative(self, x, p=0):
        """
        Derivative of link w.r.t. x
        """
        return self.links[p].link_derivative(x)

    def link_inverse_derivative(self, x, p=0):
        """
        Derivative of inverse link w.r.t. x
        """
        return self.links[p].inverse_derivative(x)

    def link_inverse_second_derivative(self, x, p=0):
        """
        Derivative of inverse link w.r.t. x
        """
        return self.links[p].inverse_second_derivative(x)
    
    def link_second_derivative(self, x, p=0):  
        return self.links[p].link_second_derivative(x)

    @staticmethod
    def _dmu(y, mu, sigma, nu, tau):
        """
        d/d(mu) of log-likelihood
        """
        z = (y - mu) / sigma
        r = nu + tau * np.arcsinh(z)
        return (z / (sigma * (z**2 + 1))) + (r * tau) / (sigma * np.sqrt(z**2 + 1))
    
    
    @staticmethod
    def _ddmu(y, mu, sigma, nu, tau):
        """
        Second derivative of log-likelihood with respect to mu.
        """
        z = (y - mu) / sigma
        r = nu + tau * np.arcsinh(z)
        t1 = (z**2 - 1) / ((z**2 + 1)**2)
        t2 = (tau**2) / (z**2 + 1)
        t3 = (tau * r * z) / ((z**2 + 1)**(3/2))
        return (t1 - t2 + t3) / (sigma**2)

    @staticmethod
    def _dsigma(y, mu, sigma, nu, tau):
        """
        d/d(sigma) of log-likelihood
        """
        z = (y - mu) / sigma
        r = nu + tau * np.arcsinh(z)
        return (-1.0 / (sigma * (z**2 + 1))) + (r * tau * z) / (sigma * np.sqrt(z**2 + 1))
    
    @staticmethod
    def _ddsigma(y, mu, sigma, nu, tau):
        """
        Second derivative of log-likelihood with respect to sigma.
        """
        z = (y - mu) / sigma
        r = nu + tau * np.arcsinh(z)
        t1 = (1 - z**2) / ((z**2 + 1)**2)
        t2 = (tau**2 * z**2) / (z**2 + 1)
        t3 = (tau * r * z * (z**2 + 2)) / ((z**2 + 1)**(3/2))
        return (t1 - t2 - t3) / (sigma**2)

    @staticmethod
    def _dnu(y, mu, sigma, nu, tau):
        """
        d/d(nu) of log-likelihood
        """
        z = (y - mu) / sigma
        r = nu + tau * np.arcsinh(z)
        return -r
    
    @staticmethod
    def _ddnu(y, mu, sigma, nu, tau):
        """
        d/d(nu) of log-likelihood
        """
        return -1

    @staticmethod
    def _dtau(y, mu, sigma, nu, tau):
        """
        d/d(tau) of log-likelihood
        """
        z = (y - mu) / sigma
        r = nu + tau * np.arcsinh(z)
        return 1.0 / tau - r* np.arcsinh(z)
    
    @staticmethod
    def _ddtau(y, mu, sigma, nu, tau):
        """
        d/d(tau) of log-likelihood
        """
        z = (y - mu) / sigma
    
        return 1.0 / tau**2 - np.arcsinh(z)**2
    
    

    def dll(self, y, theta, p):
        """
       first derivative of log-likelihood.

        """
        mu_, sigma_, nu_, tau_ = self.theta_to_params(theta)

        if p == 0:
            return self._dmu(y, mu_, sigma_, nu_, tau_)
        elif p == 1:
            return self._dsigma(y, mu_, sigma_, nu_, tau_)
        elif p == 2:
            return self._dnu(y, mu_, sigma_, nu_, tau_)
        elif p == 3:
            return self._dtau(y, mu_, sigma_, nu_, tau_)
        else:
            raise ValueError("p must be in {0,1,2,3}.")


    def _quasi_newton_score(self, y, theta, p):
        """
        Approximate second derivative as - (first_derivative^2).
        
        """
        
        dp = self.dll(y, theta, p=p)
        return -(dp * dp)
    
    
    
    def _exact_second(self, y, theta, p):
        """
        Exact second derivative
        
        """
        
        mu_, sigma_, nu_, tau_ = self.theta_to_params(theta)

        if p == 0:
            return self._ddmu(y, mu_, sigma_, nu_, tau_)
        elif p == 1:
            return self._ddsigma(y, mu_, sigma_, nu_, tau_)
        elif p == 2:
            return self._ddnu(y, mu_, sigma_, nu_, tau_)
        elif p == 3:
            return self._ddtau(y, mu_, sigma_, nu_, tau_)
        else:
            raise ValueError("p must be in {0,1,2,3}.")
        

    

    def ddll(self, y, theta, p):
        """
        Approx. second derivative of log-likelihood wrt one parameter.

        """
        sc = self._quasi_newton_score(y, theta, p)
        
        return np.clip(sc, None, -1e-15)



    def init_val(self, y):
        """
        Provide a simple initial guess for each parameter based on data.

        """

        # nu_, tau_, mu_, sigma_ = st.johnsonsu.fit(y)
        mu_ = np.mean(y) 
        sigma_ = np.std(y)
        nu_ = 0.0
        tau_ = 1

        # print(mu_, sigma_, nu_, tau_)
        # print(mu_2, sigma_2, nu_2, tau_2)
        
        params = np.array([mu_, sigma_, nu_, tau_])  

        out = np.tile(params, (y.shape[0], 1)) 
        out[:, 0] = (out[:, 0] + y)/2

        return out



    def pdf(self, y, theta):
        """
        Johnson SU pdf at y, using SciPy's johnsonsu.


        """
        mu_, sigma_, nu_, tau_ = self.theta_to_params(theta)
        return st.johnsonsu(a=nu_, b=tau_, loc=mu_, scale=sigma_).pdf(y)

    def cdf(self, y, theta):
        """
        Johnson SU cdf at y, using SciPy's johnsonsu.


        """
        mu_, sigma_, nu_, tau_ = self.theta_to_params(theta)
        return st.johnsonsu(a=nu_, b=tau_, loc=mu_, scale=sigma_).cdf(y)

 
class JSU:
    """Re‑parameterised Johnson SU distribution (µ, σ, ν, τ)

    This is the form used by **gamlss.dist::JSU**.  The transformation
    constants follow Rigby *et al.* (2019, §18.4.3).
    """


    def __init__(
        self,
        loc_link=link_id(),
        scale_link=link_log(),
        skew_link=link_id(),
        kurt_link=link_log(),
    ):
        self.n_of_p = 4
        self.loc_link = loc_link
        self.scale_link = scale_link
        self.skew_link = skew_link
        self.kurt_link = kurt_link
        self.links = [
            self.loc_link,
            self.scale_link,
            self.skew_link,
            self.kurt_link,
        ]



    def theta_to_params(self, theta):
        """Split *θ* matrix into individual parameter arrays."""
        mu, sigma, nu, tau = theta.T  # relies on theta.shape == (n,4)
        return mu, sigma, nu, tau

    def link_function(self, x, p=0):
        return self.links[p].link(x)

    def link_inverse(self, x, p=0):
        return self.links[p].inverse(x)

    def link_function_derivative(self, x, p=0):
        return self.links[p].link_derivative(x)

    def link_inverse_derivative(self, x, p=0):
        return self.links[p].inverse_derivative(x)

    def link_inverse_second_derivative(self, x, p=0):
        return self.links[p].inverse_second_derivative(x)
    def link_second_derivative(self, x, p=0):  
        return self.links[p].link_second_derivative(x)

    # ------------------------------------------------------------------


    @staticmethod
    def _common_terms(y, mu, sigma, nu, tau):
        """Return reusable intermediates needed by pdf/score/Hessian."""
        eps   = 1e-7 
        rtau = 1.0 / tau                      
        w     = np.clip(np.exp(rtau**2), 1.0 + eps, None)
        omega = np.clip(-nu * rtau, -300, 300)
        c = (0.5 * (w - 1.0) * (w * np.cosh(2.0 * omega) + 1.0))**(-0.5)
        z = (y - (mu + c * sigma * np.sqrt(w) * np.sinh(omega))) / (c * sigma)
        r = -nu + np.arcsinh(z) / rtau
        
        return rtau, w, omega, c, z, r



    @classmethod
    def _dmu(cls, y, mu, sigma, nu, tau):
        rtau, w, omega, c, z, r = cls._common_terms(y, mu, sigma, nu, tau)
        return (z / (z * z + 1.0) + r / (rtau * np.sqrt(z * z + 1.0))) / (c * sigma)

    @classmethod
    def _dsigma(cls, y, mu, sigma, nu, tau):
        rtau, w, omega, c, z, r = cls._common_terms(y, mu, sigma, nu, tau)
        term = (z + np.sqrt(w) * np.sinh(omega)) * (
            z / (z * z + 1.0) + r / (rtau * np.sqrt(z * z + 1.0))
        )
        return (term - 1.0) / sigma

    @classmethod
    def _dnu(cls, y, mu, sigma, nu, tau):
        rtau, w, omega, c, z, r = cls._common_terms(y, mu, sigma, nu, tau)
        dlogcdv = (rtau * w * np.sinh(2.0 * omega)) / (w * np.cosh(2.0 * omega) + 1.0)
        dzdv = (
            -(z + np.sqrt(w) * np.sinh(omega)) * dlogcdv
            + rtau * np.sqrt(w) * np.cosh(omega)
        )
        score = -dlogcdv - (
            z / (z * z + 1.0) + r / (rtau * np.sqrt(z * z + 1.0))
        ) * dzdv + r
        return score

    @classmethod
    def _dtau(cls, y, mu, sigma, nu, tau):
        rtau, w, omega, c, z, r = cls._common_terms(y, mu, sigma, nu, tau)
        dlogcdt = (
            -rtau
            * w
            * ((1.0 / (w - 1.0)) + (np.cosh(2.0 * omega) / (w * np.cosh(2.0 * omega) + 1.0)))
        ) + (nu * w * np.sinh(2.0 * omega)) / (w * np.cosh(2.0 * omega) + 1.0)

        dzdt = (
            -(z + np.sqrt(w) * np.sinh(omega)) * dlogcdt
            - rtau * np.sqrt(w) * np.sinh(omega)
            + nu * np.sqrt(w) * np.cosh(omega)
        )

        inner = (
            z / (z * z + 1.0) + r / (rtau * np.sqrt(z * z + 1.0))
        ) * dzdt
        dldt = -dlogcdt - (1.0 / rtau) - inner + (r * (r + nu)) / rtau
        return -dldt * (rtau ** 2)


    @staticmethod
    def _quasi_newton_score(d1):
        """Return −(∂ℓ/∂θ)² with a small negative floor."""
        return np.clip(-d1 * d1, None, -1e-15)


    def dll(self, y, theta, p):
        mu, sigma, nu, tau = self.theta_to_params(theta)
        if p == 0:
            return self._dmu(y, mu, sigma, nu, tau)
        elif p == 1:
            return self._dsigma(y, mu, sigma, nu, tau)
        elif p == 2:
            return self._dnu(y, mu, sigma, nu, tau)
        elif p == 3:
            return self._dtau(y, mu, sigma, nu, tau)
        else:
            raise ValueError("p must be in {0,1,2,3}.")

    def ddll_(self, y, theta, p):
        d1 = self.dll(y, theta, p)
        return self._quasi_newton_score(d1)

    def ddll(self, y, theta, p):
        mu, sigma, nu, tau = self.theta_to_params(theta)
        if p == 0:
            dmu = self._dmu(y, mu, sigma, nu, tau)
            return np.clip(-dmu * dmu, None, -1e-15)
        elif p == 1:
            dsigma = self._dsigma(y, mu, sigma, nu, tau)
            return np.clip(-dsigma * dsigma, None, -1e-15)
        elif p == 2:
            dnu = self._dnu(y, mu, sigma, nu, tau)
            return np.clip(-dnu * dnu, None, -1e-4)
        elif p == 3:
            dtau = self._dtau(y, mu, sigma, nu, tau)
            return np.clip(-dtau * dtau, None, -1e-4)
        else:
            raise ValueError("p must be in {0,1,2,3}.")
        

    def pdf(self, y, theta):
        mu, sigma, nu, tau = self.theta_to_params(theta)
        rtau, w, omega, c, z, r = self._common_terms(y, mu, sigma, nu, tau)
    
        denom = c * sigma * np.sqrt(z * z + 1) * (2.0 * np.pi)**(0.5)
        pdf = tau / denom * np.exp(- 0.5 * r * r )
        return pdf

    def cdf(self, y, theta):
        mu, sigma, nu, tau = self.theta_to_params(theta)
        rtau, w, omega, c, z, r = self._common_terms(y, mu, sigma, nu, tau)
        return st.norm.cdf(r)
    
    @classmethod
    def moments(cls, mu, sigma, nu, tau):

        mean = mu
        std  = sigma

        omega = np.exp(1 / (tau**2))
        c = 1/np.sqrt( 1/2* (omega -1) * (omega* np.cosh(2*nu/tau) + 1) )
        mu3 = 1/4 * np.power(c, 3) * np.power(sigma, 3) * np.sqrt(omega) * np.power(omega-1, 2) * ( omega * (omega + 2) * np.sinh(3 * nu/tau ) + 3*np.sinh(nu/tau))
        skew = mu3/np.power(sigma, 3)

        mu4 = 1/8 * np.power(c, 4) * np.power(sigma, 4) * np.power(omega-1, 2) * (
            np.power(omega, 2) * (np.power(omega, 4) + 2 * np.power(omega, 3) + 3*np.power(omega, 2) -3) * np.cosh(4*nu/tau) +
            4* np.power(omega, 2) * (omega + 2) * np.cosh(2*nu/tau) + 3* (2*omega + 1)
        )
        excess_kurtosis = mu4/(np.power(sigma, 4)) - 3

        return mean, std, skew, excess_kurtosis



    def init_val(self, y):
        """Very crude JSU starting values (mean/SD & default shapes)."""
        # nu_, tau_, mu_, sigma_ = st.johnsonsu.fit(y)
        # mu_, sigma_, nu_, tau_ = jsu_original_to_reparam(mu_, sigma_, nu_, tau_)

        mu_ = np.mean(y) 
        sigma_ = np.std(y)/4
        
        nu_ = 0.0
        tau_ = 1

        
        params = np.array([mu_, sigma_, nu_, tau_])  

        out = np.tile(params, (y.shape[0], 1)) 
        out[:, 0] = (out[:, 0] + y)/2
        return out
