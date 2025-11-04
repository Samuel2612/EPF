import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.stats import norm
import matplotlib.pyplot as plt



def phi_st_gbm(omega, S0, r, sigma, T, M=100):
    """
    Integration performed with Gauss–Hermite quadrature (degree M).
    """
    mu_ln = np.log(S0) + (r - 0.5 * sigma ** 2) * T            
    nodes, weights = hermgauss(M)                           
    z = np.sqrt(2) * nodes                                   
    s_vals = np.exp(mu_ln + sigma * np.sqrt(T) * z)           
    cf_val = np.sum(weights * np.exp(1j * omega * s_vals)) / np.sqrt(np.pi)
    return cf_val



def psi_k(k, c, d, a, b):
    if k == 0:
        return d - c
    return (b - a) / (k * np.pi) * (
        np.sin(k * np.pi * (d - a) / (b - a)) -
        np.sin(k * np.pi * (c - a) / (b - a))
    )


def xi_k(k, c, d, a, b):
    if k == 0:
        return 0.5 * (d ** 2 - c ** 2)
    term = (b - a) / (k * np.pi)
    return term * (
        d * np.sin(k * np.pi * (d - a) / (b - a)) -
        c * np.sin(k * np.pi * (c - a) / (b - a))
    ) + term ** 2 * (
        np.cos(k * np.pi * (d - a) / (b - a)) -
        np.cos(k * np.pi * (c - a) / (b - a))
    )



def cos_price_ratio(S0, K, r, T, sigma,
                    N=256, L=10, M=100, option_type='call'):

    c1 = (r - 0.5 * sigma ** 2) * T + np.log(S0 / K)  
    c2 = sigma ** 2 * T                             
    c4 = 3 * sigma ** 4 * T                            # 4th cumulant
    a = 0.0
    b = np.exp(c1 + L * np.sqrt(c2 + np.sqrt(c4)))     # Fang-Oosterlee


    V = np.zeros(N)
    for k in range(N):
        if option_type == 'call':          # payoff = K·max(Y−1,0)
            V[k] = 2 * K / (b - a) * (xi_k(k, 1, b, a, b) - psi_k(k, 1, b, a, b))
        else:                              # payoff = K·max(1−Y,0)
            V[k] = 2 * K / (b - a) * (-xi_k(k, a, 1, a, b) + psi_k(k, a, 1, a, b))


    total = 0.0
    for k in range(N):
        omega = k * np.pi / (K * (b - a))            
        phi = phi_st_gbm(omega, S0, r, sigma, T, M)
        term = np.real(phi * np.exp(-1j * k * np.pi * a / (b - a))) * V[k]
        if k == 0:
            term *= 0.5                               
        total += term

    return np.exp(-r * T) * total



def bs_prices(S0, K, r, T, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put  = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return call, put



def colour_axis(ax, colour):
    ax.spines['left' if ax.yaxis.get_label_position() == 'left' else 'right'
              ].set_color(colour)
    ax.tick_params(axis='y', colors=colour)
    ax.yaxis.label.set_color(colour)


if __name__ == "__main__":
    S0, r, T, sigma = 100.0, 0.05, 1.0, 0.20
    K_vals = np.linspace(50, 150, 51)

    call_cos, call_bs, put_cos, put_bs = [], [], [], []
    for K in K_vals:
        call_cos.append(cos_price_ratio(S0, K, r, T, sigma, option_type='call'))
        put_cos.append(cos_price_ratio(S0, K, r, T, sigma, option_type='put'))
        c_bs, p_bs = bs_prices(S0, K, r, T, sigma)
        call_bs.append(c_bs)
        put_bs.append(p_bs)

    call_cos, call_bs = np.array(call_cos), np.array(call_bs)
    put_cos,  put_bs  = np.array(put_cos),  np.array(put_bs)

    call_err = call_cos - call_bs
    put_err  = put_cos  - put_bs

    fig1, ax_price = plt.subplots(figsize=(15, 9))
    # ax_price.plot(K_vals, call_cos, color='black', lw=1.5, label='COS price')
    ax_price.plot(K_vals, call_bs,  color='green', ls='--', lw=4.0,
                  label='BS price')
    ax_price.plot(K_vals, call_cos, 'o:b', label='COS price')
    ax_price.set_xlabel("Strike K")
    ax_price.set_ylabel("Call price")
    colour_axis(ax_price, 'black')

    ax_err = ax_price.twinx()
    ax_err.plot(K_vals, put_err, '.:r', label='Error (COS-BS)')
    ax_err.set_ylabel("Price difference")
    colour_axis(ax_err, 'red')

    lines, labels = ax_price.get_legend_handles_labels()
    lines2, labels2 = ax_err.get_legend_handles_labels()
    ax_price.legend(lines + lines2, labels + labels2,
                    loc="upper center", ncol=3, fontsize=9)
    ax_price.set_title("Call options – COS vs Black-Scholes")
    plt.tight_layout()


    fig2, ax_price2 = plt.subplots(figsize=(15, 9))
    ax_price2.plot(K_vals, put_bs,  color='green', ls='--', lw=4.0,
                   label='BS price')
    ax_price2.plot(K_vals, put_cos, 'o:b', label='COS price')
    ax_price2.set_xlabel("Strike K")
    ax_price2.set_ylabel("Put price")
    colour_axis(ax_price2, 'green')

    ax_err2 = ax_price2.twinx()
    ax_err2.plot(K_vals, put_err, '.:r', label='Error (COS-BS)')
    ax_err2.set_ylabel("Price difference")
    colour_axis(ax_err2, 'red')

    lines, labels = ax_price2.get_legend_handles_labels()
    lines2, labels2 = ax_err2.get_legend_handles_labels()
    ax_price2.legend(lines + lines2, labels + labels2,
                     loc="upper center", ncol=3, fontsize=9)
    ax_price2.set_title("Put options – COS vs Black-Scholes")
    plt.tight_layout()

    plt.show()