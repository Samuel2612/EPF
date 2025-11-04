import numpy as np
from costt import tt_svd, build_tensor


def cos_coeff_std_norm(K, a, b, M=4001):
    z = np.linspace(a, b, M)
    pdf = np.exp(-0.5 * z**2) / np.sqrt(2*np.pi)
    fac = np.pi/(b-a)
    G = np.empty(K+1)
    for k in range(K+1):
        G[k] = 2/(b-a)*np.trapz(pdf*np.cos(k*fac*(z-a)), z)
    return G

def build_tensor_(G_list):
    A = G_list[0]
    for G in G_list[1:]:
        A = np.multiply.outer(A, G)          # rank-1 outer product
    return A

def I_jk(ω, wj,S0j, μj,σj, k, a,b, z, cos_cache):
    f = np.exp(μj+σj*z)
    g = np.exp(1j*ω*wj*S0j*f)*cos_cache[k]
    return np.trapz(g, z)

def φ_H_tt(ω_vec, cores, params, prime, K, a,b, M=4001):
    z   = np.linspace(a,b,M)
    fac = np.pi/(b-a)
    cos_cache = np.array([np.cos(k*fac*(z-a)) for k in range(K+1)])
    n   = len(params)

    φ = np.empty(len(ω_vec), dtype=complex)
    for t,ω in enumerate(ω_vec):
        mat = np.array([[1.+0j]])
        for j,(wj,S0j,μj,σj) in enumerate(params):
            I = np.array([I_jk(ω,wj,S0j,μj,σj,k,a,b,z,cos_cache)
                          for k in range(K+1)])
            v = prime*I                                   # length K+1
            core = cores[j]                               # (r_{j-1},K+1,r_j)
            Mj = np.tensordot(core, v, axes=[1,0])        # (r_{j-1}, r_j)
            mat = mat @ Mj
        φ[t] = mat[0,0]
    return φ

def V_k_num(k, K_strike, A,B, kind="call", N=20001):
    h = np.linspace(A,B,N)
    f = np.maximum(h-K_strike,0) if kind=="call" else np.maximum(K_strike-h,0)
    c = np.cos(k*np.pi*(h-A)/(B-A))
    return 2/(B-A)*np.trapz(f*c, h)
# --------------------------------------------------  pricer ---------
def basket_cos_tt(S0, w, σ, r,T, K_strike,
                  kind="call", K_terms=40, N_cos=256,
                  L_z=8., L_H=8., M_int=4001, tt_eps=1e-10):
    S0,w,σ = map(np.asarray,(S0,w,σ))
    n      = len(S0)
    Ks = [K_terms]**n

    # 1-D cosine coeffs  G_j(k)
    a,b = -L_z, +L_z
    G_list = [cos_coeff_std_norm(K_terms,a,b,M_int) for _ in range(n)]

    # coefficient tensor  →  TT cores
    A_full      = build_tensor_lowmem(Ks, a, b, mu, Sigma)
    cores,_     = tt_svd(A_full, eps=tt_eps)

    # μ_j , σ_j√T  in log-normal
    μ  = (r-0.5*σ**2)*T
    σT = σ*np.sqrt(T)
    params = [(w[j],S0[j], μ[j], σT[j]) for j in range(n)]

    # prime weight
    prime = np.ones(K_terms+1); prime[0]=0.5

    # payoff interval  [A,B]   =  [0 , mean + L_H·std]
    ES   = S0*np.exp(μ+0.5*σT**2)
    VarS = S0**2*np.exp(2*μ+σT**2)*(np.exp(σT**2)-1)
    mean,std = np.dot(w,ES), np.sqrt(np.dot(w**2,VarS))
    A_pay, B_pay = 0., mean + L_H*std

    # outer COS sum
    k  = np.arange(N_cos)
    u  = k*np.pi/(B_pay-A_pay)
    φ  = φ_H_tt(u, cores, params, prime, K_terms, a,b, M_int)
    V  = np.array([V_k_num(ki,K_strike,A_pay,B_pay,kind) for ki in k])

    weight = np.ones(N_cos); weight[0]=0.5
    price  = np.exp(-r*T)*np.sum(weight*np.real(φ*V))
    return float(price)
# --------------------------------------------------  demo -----------
if __name__ == "__main__":
    S0     = [100, 120]
    w      = [0.6, 0.4]
    σ      = [0.20, 0.25]
    r,T    = 0.05, 1.0
    K_strk = 110

    px = basket_cos_tt(S0,w,σ,r,T,K_strk,
                       kind="call",
                       K_terms=40, N_cos=256,
                       L_z=8, L_H=8,
                       M_int=4001, tt_eps=1e-10)
    print(f"COS-TT basket-call ≈ {px:.6f}")
