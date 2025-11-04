# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:56:33 2025

@author: samue
"""

import numpy as np
import scipy.stats as st



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



class link_id():
    """
    The identity link function.

    """

    def __init__(self):
        pass

    def link(self, x: np.ndarray) -> np.ndarray:
        return x

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)
    
    def inverse_second_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)






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
        sc = self._quasi_newton_score(y, theta, p=p)
        
        return np.clip(sc, None, -1e-15)



    def init_val(self, y):
        """
        Provide a simple initial guess for each parameter based on data.

        """

        nu_, tau_, mu_, sigma_ = st.johnsonsu.fit(y)
        # mu_2 = np.mean(y)
        # sigma_2 = np.std(y)
        # nu_2 = 0.0
        # tau_2 = 2

        # print(mu_, sigma_, nu_, tau_)
        # print(mu_2, sigma_2, nu_2, tau_2)
        
        params = np.array([mu_, sigma_, nu_, tau_])  

        out = np.tile(params, (y.shape[0], 1)) 

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

 
class JSU2:
    """Re‑parameterised Johnson SU distribution (mu, sigma, nu, tau)

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
        """Split *theta* matrix into individual parameter arrays."""
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
    
    
    @staticmethod
    def _safe_div(a,b):
        return a/np.where(np.abs(b)<1e-15, np.sign(b)*1e-15, b)


    @staticmethod
    def _common_terms(y, mu, sigma, nu, tau):
        """Return reusable intermediates needed by pdf/score/Hessian."""
        rtau = 1.0 / tau
        w = np.where(rtau < 1e-7, 1.0, np.exp(rtau**2))
        omega = -nu * rtau
        c = (0.5 * (w - 1.0) * (w * np.cosh(2.0 * omega) + 1.0)) ** (-0.5)
        z = (y - (mu + c * sigma * np.sqrt(w) * np.sinh(omega))) / (c * sigma)
        r = -nu + np.arcsinh(z) / rtau
        return rtau, w, omega, c, z, r


    @staticmethod
    def _common_terms1(y, mu, sigma, nu, tau):
        """Return reusable intermediates needed by pdf/score/Hessian."""
        eps   = 1e-7 
        # eps_sqr = 1e-10
        rtau = 1.0 / tau                      
        w     = np.clip(np.exp(rtau**2), 1.0 + eps, None)
        omega = np.clip(-nu * rtau, -300, 300)
        # omega = -nu * rtau
        # term = np.clip((w - 1.0)**(0.5), eps_sqr, None)
        # term = (w - 1.0)**(0.5)
        # denom = (0.5  * (w * np.cosh(2.0 * omega) + 1.0))**(0.5)*term
        c = (0.5 * (w - 1.0) * (w * np.cosh(2.0 * omega) + 1.0))**(-0.5)
        # c = 1/denom
        z = (y - (mu + c * sigma * np.sqrt(w) * np.sinh(omega))) / (c * sigma)
        r = -nu + np.arcsinh(z) / rtau
        
        return rtau, w, omega, c, z, r

    @staticmethod
    def _D(z):
        return z * z + 1.0


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
        """Return −(∂ℓ/∂theta)² with a small negative floor."""
        return np.clip(-d1 * d1, None, -1e-15)


    @classmethod
    def _ddmu(cls, y, mu, sigma, nu, tau):
        rtau,w,omega,c,z,r = cls._common_terms(y,mu,sigma,nu,tau)
        D           = cls._D(z)
        out = ((z*z - 1.0)/(D*D) - (tau*tau)/D + (tau*r*z)/(D**1.5)) / ((c*sigma)**2)
        return out


    @classmethod
    def _ddsigma(cls, y, mu, sigma, nu, tau):
        rtau, w, omega, c, z, r = cls._common_terms(y, mu, sigma, nu, tau)
        D  = z*z + 1.0
        S  = np.sqrt(w)*np.sinh(omega)
        A  = z + S
        B  = z/D + r/(rtau*np.sqrt(D))
    
        dz   = -A / sigma                       # ∂z/∂σ
        dr   = dz / (rtau*np.sqrt(D))           # ∂r/∂σ
    
        dB   = (dz/D - 2*z*z*dz/D**2) \
             + (dr/(rtau*np.sqrt(D)) - r*z*dz/(rtau*D**1.5))
    
        dterm = dz*B + A*dB
        return dterm/sigma - (A*B - 1.0)/sigma**2



    

    @classmethod
    def _dlogc_dnu(cls, rtau, w, omega):
        return cls._safe_div(rtau*w*np.sinh(2*omega),(w*np.cosh(2*omega)+1.0))

    @classmethod
    def _d2logc_dnu2(cls, rtau, w, omega):
        denom = w*np.cosh(2*omega)+1.0
        num   = rtau*w*np.sinh(2*omega)
        dnum  = -2*rtau*rtau*w*np.cosh(2*omega)
        dden  = -2*rtau*w*np.sinh(2*omega)
        return cls._safe_div((dnum*denom - num*dden),(denom*denom))

    @classmethod
    def _ddnu(cls, y, mu, sigma, nu, tau):
        rtau, w, omega, c, z, r = cls._common_terms(y, mu, sigma, nu, tau)
        D      = z*z + 1.0
        S      = np.sqrt(w)*np.sinh(omega)
    
        dlogc  = cls._dlogc_dnu(rtau, w, omega)
        d2logc = cls._d2logc_dnu2(rtau, w, omega)
    
        dz     = -(z+S)*dlogc + rtau*np.sqrt(w)*np.cosh(omega)
        dz2    = -(z+S)*d2logc - dz*dlogc                 # ∂²z/∂ν²
    
        dr     = -1.0 + tau*dz/np.sqrt(D)
        dr2    = tau*(dz2/np.sqrt(D) - 0.5*z*dz*dz/D**1.5)
    
        term   = z/D + r/(rtau*np.sqrt(D))
        dterm  = (dz/D - 2*z*z*dz/D**2) \
               + (dr/(rtau*np.sqrt(D)) - r*z*dz/(rtau*D**1.5))
    
        return -d2logc - dterm*dz - term*dz2 + dr2


    @classmethod
    def _dlogc_dtau(cls, rtau, w, omega, nu):
        denom = w*np.cosh(2*omega)+1.0
        g1    = cls._safe_div(1.0,(w-1.0))
        return -rtau*w*(g1+np.cosh(2*omega)/denom) + nu*w*np.sinh(2*omega)/denom

    @classmethod
    def _d2logc_dtau2(cls, rtau, w, omega, nu):
        # manual analytic derivative (vectorised)
        drtau   = -rtau*rtau
        dw    = -2*w*rtau*rtau*rtau
        domeg = nu*rtau*rtau
        sinh2 = np.sinh(2*omega)
        cosh2 = np.cosh(2*omega)
        dcosh2= 2*sinh2*domeg
        dsinh2= 2*cosh2*domeg
        denom = w*cosh2 + 1.0
        dden  = dw*cosh2 + w*dcosh2

        g1    = cls._safe_div(1.0,(w-1.0))          
        dg1   = cls._safe_div(-dw,((w-1.0)**2))
        g2    = cosh2/denom
        dg2   = (dcosh2*denom - cosh2*dden)/(denom*denom)

        part1 = -rtau*w*(dg1+dg2) - drtau*w*(g1+g2) - rtau*dw*(g1+g2)
        part2 = nu*(dw*sinh2/denom + w*dsinh2/denom - w*sinh2*dden/(denom*denom))
        return part1 + part2

    @classmethod
    def _ddtau(cls, y, mu, sigma, nu, tau):
        rtau, w, omega, c, z, r = cls._common_terms(y, mu, sigma, nu, tau)
        D      = z*z + 1.0
        S      = np.sqrt(w)*np.sinh(omega)
    
        dlogc  = cls._dlogc_dtau(rtau, w, omega, nu)
        d2logc = cls._d2logc_dtau2(rtau, w, omega, nu)
    
        dz     = -(z+S)*dlogc - rtau*S + nu*np.sqrt(w)*np.cosh(omega)
        dz2    = -(z+S)*d2logc - dz*dlogc + 2*rtau*rtau*S   
    
        dr     = -(r+nu)/tau + tau*dz/np.sqrt(D)
        dr2    = (-2.0*dr/tau - 2.0*(r+nu)/tau**2
                  + tau*(dz2/np.sqrt(D) - 0.5*z*dz*dz/D**1.5))
    
        term   = z/D + r/(rtau*np.sqrt(D))
        dterm  = (dz/D - 2*z*z*dz/D**2) \
               + (dr/(rtau*np.sqrt(D)) - r*z*dz/(rtau*D**1.5))
    
        return -(d2logc + 2.0/tau**2) - dterm*dz - term*dz2 + dr2



    # ------------------------------------------------------------------
    #  public wrappers --------------------------------------------------
    # ------------------------------------------------------------------

    def dll(self, y, theta, p):
        mu,sigma,nu,tau = self.theta_to_params(theta)
        if   p==0: return self._dmu(y,mu,sigma,nu,tau)
        elif p==1: return self._dsigma(y,mu,sigma,nu,tau)
        elif p==2: return self._dnu(y,mu,sigma,nu,tau)
        elif p==3: return self._dtau(y,mu,sigma,nu,tau)
        else: raise ValueError("p must be 0…3")

    def ddll(self, y, theta, p):
        mu,sigma,nu,tau = self.theta_to_params(theta)
        if   p==0: val = self._ddmu(y,mu,sigma,nu,tau)
        elif p==1: val = self._ddsigma(y,mu,sigma,nu,tau)
        elif p==2: val = self._ddnu(y,mu,sigma,nu,tau)
        elif p==3: val = self._ddtau(y,mu,sigma,nu,tau)
        else: raise ValueError("p must be 0…3")
        return np.clip(val, None, -1e-15)  # numerical safety



    def pdf(self, y, theta):
        mu, sigma, nu, tau = self.theta_to_params(theta)
        rtau, w, omega, c, z, r = self._common_terms(y, mu, sigma, nu, tau)
        logpdf = (
            -np.log(sigma)
            - np.log(c)
            - np.log(rtau)
            - 0.5 * np.log(z * z + 1.0)
            - 0.5 * np.log(2.0 * np.pi)
            - 0.5 * r * r
        )
        return np.exp(logpdf)

    def cdf(self, y, theta):
        mu, sigma, nu, tau = self.theta_to_params(theta)
        rtau, w, omega, c, z, r = self._common_terms(y, mu, sigma, nu, tau)
        return st.norm.cdf(r)

    # ------------------------------------------------------------------
    #  Initial values
    # ------------------------------------------------------------------

    def init_val(self, y):
        """Very crude *JSU* starting values (mean/SD & default shapes)."""
        mu_ = np.mean(y) 
        sigma_ = np.std(y)/4
        nu_ = 0.0
        tau_ = 1

        
        params = np.array([mu_, sigma_, nu_, tau_])  

        out = np.tile(params, (y.shape[0], 1)) 
        return out
