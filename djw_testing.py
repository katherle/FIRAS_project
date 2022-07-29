import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats
import emcee as mc
from scipy.optimize import curve_fit

from astropy import units as u
from astropy import constants as const
from astropy.table import QTable
from astropy.visualization import quantity_support
from astropy.modeling.models import BlackBody
from astropy.io import fits

quantity_support()
set_matplotlib_formats('svg')
from matplotlib import cm
from cycler import cycler
plt.rc('legend', frameon = False)
#plt.rc('figure', figsize = (7, 7/1.25))
#plt.rc('font', size = 14)

#get the FIRAS data from their paper:
names = ["freq", "monopole", "res", "sigma", "gal"]
tmp = pd.read_csv("https://lambda.gsfc.nasa.gov/data/cobe/firas/monopole_spec/firas_monopole_spec_v1.txt", delim_whitespace = True, header = 0, skiprows = 17, names = names)
unit_names = ["cm-1", "MJy/sr", "kJy/sr", "kJy/sr", "kJy/sr"]
units = {names[i]:unit_names[i] for i in range(len(names))}
firas = QTable.from_pandas(tmp, units = units)

def func(nu, T):
    #nu and T must both be in SI units
    h = const.h.value
    c = const.c.value
    kb = const.k_B.value
    return (2*h*nu**3/c**2 / (np.exp(h*nu/kb/T) - 1))

### also try with data from Fixsen (1996):
firas_freq = firas["freq"].to('GHz', equivalencies = u.spectral())
#bb_firas = BlackBody(2.728*u.K)
#firas["monopole"] = bb_firas(firas_freq) + firas["res"]

T_simple, pcov_simple = curve_fit(func, xdata = firas_freq.si.value, ydata = firas["monopole"].si.value)
print(T_simple)
#print(firas_freq)

bb_simple = BlackBody(T_simple*u.K)
plt.figure()
plt.errorbar(firas_freq, firas["monopole"], yerr = firas["sigma"], fmt = 'k.')
plt.plot(firas_freq, bb_simple(firas_freq).to('MJy/sr'), color = 'xkcd:turquoise', ls = "--", label = "fit")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Intensity (MJy/sr)")
plt.title("Temperature fit")
plt.legend()
plt.show()



def dBdT(nu, T = 2.725*u.K):
    #we want nu in s-1
    #all constants in SI units
    x = (const.h*nu/const.k_B/T).to('').value
    return 2 * const.h**2 * nu**4 / (const.k_B * const.c**2 * T**2) * np.exp(x) / np.expm1(x)**2

def func_mu(nu, mu):
    #nu in GHz (but without astropy units or scipy freaks out), T in K
    #all constants in SI units
    nu = (nu*u.GHz).to('s-1')
    T = 2.725*u.K
    x = const.h*nu/const.k_B/T
    #return (-T*mu/x * dBdT(nu)).to('kJy').value
    x = x.to('').value
    I_0 = 270*u.MJy
    beta = 2.1923
    return -(I_0*x**4*np.exp(x)/np.expm1(x)**2*(1/beta - 1/x)*mu).to('kJy').value


popt_mu, pcov_mu = curve_fit(func_mu, xdata = firas_freq.value, ydata = firas["res"])
print(popt_mu)

plt.figure()
plt.plot(firas_freq, firas["res"], "k", lw = 1)
plt.plot(firas_freq, (func_mu(firas_freq.value, popt_mu[0]))*u.kJy/u.sr, color = 'xkcd:turquoise', ls = "--", label = "fit")
plt.ylim([-100, 100])
plt.xlabel("Frequency (GHz)")
plt.ylabel("Residual intensity (kJy/sr)")
plt.title(r"Temperature + $\mu$")
plt.tight_layout()
plt.legend()
plt.show()


def func_y(nu, y):
    #nu in GHz (but without astropy units or scipy freaks out), T in K
    #all constants in SI units
    #note: having problems with overflow errors
    nu = (nu*u.GHz).to('s-1')
    T = T_simple*u.K
    x = const.h*nu/const.k_B/T
    #note: typo in F96, compare to Bianchini & Fabbian 2022
    return (y*T*(x*(np.exp(x)+1)/np.expm1(x) - 4)*dBdT(nu)).to('kJy').value

popt_y, pcov_y = curve_fit(func_y, xdata = firas_freq.value, ydata = firas["res"])
print(popt_y)

plt.figure()
plt.plot(firas_freq, firas["res"], "k", lw = 1)
plt.plot(firas_freq, func_y(firas_freq.value, popt_y[0])*u.kJy/u.sr, color = 'xkcd:turquoise', ls = "--", label = "fit")
plt.ylim([-100, 100])
plt.xlabel("Frequency (GHz)")
plt.ylabel("Residual intensity (kJy/sr)")
plt.title("Temperature + y")
plt.legend()
plt.show()

plt.figure()
plt.plot(firas_freq, firas["res"], "k", lw = 1)
plt.plot(firas_freq, (func_mu(firas_freq.value, -9e-5))*u.kJy/u.sr, color = 'xkcd:turquoise', ls = "--", label = "fit (mu)")
plt.plot(firas_freq, (func_y(firas_freq.value, 15e-6)*u.kJy/u.sr), color = 'xkcd:orange', ls = "-.", label = "fit (y)")
plt.plot(firas_freq, 0.0001*u.K*dBdT(firas_freq.to("s-1"))/u.sr, label = '0.0001 dB/dT')
plt.ylim([-100, 100])
plt.xlabel("Frequency (GHz)")
plt.ylabel("Residual intensity (kJy/sr)")
plt.title("Reproducing Fig. 5 (Fixsen et al 1996)")
plt.tight_layout()
plt.legend()
plt.show()



def log_Ly(theta, nu, Sres, Serr):
    #nu is an array of frequencies in s-1
    #Sres and Serr are the residual and error arrays in kJy/sr
    T, y = theta
    #T = theta
    #print(T)
    #y = -1e-6
    x = const.h*nu/const.k_B/(T*u.K)
    #print(np.exp(x))
    model = (y*T*u.K * (x * (np.exp(x)+1) / np.expm1(x) - 4) * dBdT(nu, T*u.K)).to('kJy')/u.sr
    #print(model)
    log_L = -0.5 * np.sum((Sres-model)**2/Serr**2)
    #print(log_L)
    return log_L


def log_Lmu(theta, nu, Sres, Serr):
    T, mu = theta
    x = const.h*nu/const.k_B/(T*u.K)
    model = func_mu(nu/u.GHz, mu)/u.sr*u.kJy
    log_L = -0.5 * np.sum((Sres-model)**2/Serr**2)
    #print(T)
    #print(log_L)
    return log_L

def log_LBB(T, nu, S, Serr):
    #print(T)
    #bb = BlackBody(T*u.K)
    #model = bb(nu).to('MJy/sr')
    x = const.h*nu/const.k_B/(T*u.K)
    x = x.to('').value
    model = 2*const.h*nu**3/const.c**2/np.expm1(x)/u.sr

    model_v = model.to('MJy/sr').value
    S_v = S.to('MJy/sr').value
    Serr_v = Serr.to('MJy/sr').value


    log_L = -0.5 * np.sum((S_v-model_v)**2/Serr_v**2)
    #print(log_L)
    return log_L

def log_Ltot(theta, nu, S, Sres, Serr):
    T, y, mu = theta
    return log_LBB(T, nu, S, Serr) + \
           log_Ly(np.array([T,y]), nu, Sres, Serr) + \
           log_Lmu(np.array([T,mu]), nu, Sres, Serr)


#maximum likelihood:
from scipy.optimize import minimize

nll_y = lambda *args: -log_Ly(*args)
nll_mu = lambda *args: -log_Lmu(*args)
nll_BB = lambda *args: -log_LBB(*args)
nll_tot = lambda *args: -log_Ltot(*args)
T_ini = 2.728
#y_ini = -1e-6
#mu_ini = -1e-5
y_ini = -1
mu_ini = -1
initial_y = np.array([T_ini, y_ini])
initial_mu = np.array([T_ini, mu_ini])
initial = np.array([T_ini])
initial_tot = np.array([T_ini, y_ini, mu_ini])
print("Initial:")
print(T_ini)
print(y_ini)
print(mu_ini)

soln_y = minimize(nll_y, initial_y, args = (firas_freq.to('s-1')[0:-3], firas["res"][0:-3], firas["sigma"][0:-3]), bounds = [(1, 3), (None, None)])
soln_mu = minimize(nll_mu, initial_mu, args = (firas_freq.to('s-1')[0:-3], firas["res"][0:-3], firas["sigma"][0:-3]), bounds = [(1, 3), (None, None)])
soln_BB = minimize(nll_BB, initial, args = (firas_freq.to('s-1')[0:-3], firas["monopole"][0:-3], firas["sigma"][0:-3]), bounds = [(1, 3)])
soln_tot = minimize(nll_tot, initial_tot, args = (firas_freq.to('s-1')[0:-3], firas["monopole"][0:-3], firas["res"][0:-3], firas["sigma"][0:-3]), bounds = [(None, None), (None, None), (None, None)])
T_ml_y, y_ml = soln_y.x
T_ml_mu, mu_ml = soln_mu.x
T_ml = soln_BB.x
print('Total ML')
x_tot = soln_tot.x

print("Maximum likelihood estimates:")
print(T_ml_y, ",  ", T_ml_mu, ",  ", T_ml)
print(y_ml)
print(mu_ml)

plt.figure()
plt.errorbar(firas_freq.to('s-1')[0:-3], firas["res"][0:-3], yerr = firas["sigma"][0:-3], fmt = 'k.', elinewidth = 1)
x = const.h*firas_freq.to('s-1')[0:-3]/const.k_B/(T_ml_y*u.K)
model_mu = (-T_ml_mu*u.K*mu_ml/x * dBdT(firas_freq.to('s-1')[0:-3])).to('kJy')/u.sr
model_y = (y_ml*T_ml_y*u.K*(x*(np.exp(x)+1)/np.expm1(x) - 4)*dBdT(firas_freq.to('s-1')[0:-3])).to('kJy')/u.sr
plt.plot(firas_freq.to('s-1')[0:-3], model_mu, lw = 1, ls = '--', color = 'xkcd:turquoise')
plt.plot(firas_freq.to('s-1')[0:-3], model_y, lw = 1, ls = '--', color = 'xkcd:orange')
#plt.ylim([-100, 100])
plt.grid()
plt.title('Maximum likelhood fits')
plt.tight_layout()
plt.show()

plt.figure()
plt.errorbar(firas_freq.to('s-1'), firas["monopole"], yerr = firas["sigma"], fmt = 'k.', elinewidth = 1)
model = BlackBody(T_ml*u.K)
plt.plot(firas_freq.to('s-1'), model(firas_freq).to('MJy/sr'), lw = 1, ls = '--', color = 'xkcd:turquoise')
plt.grid()
plt.title('Maximum likelihood Monopole')
plt.tight_layout()
plt.show()

plt.figure()
T = np.linspace(1, 3, 100)
T = np.linspace(2.72, 2.73, 1000)
likelihood_BB = np.zeros_like(T)
for i in range(len(T)):
    likelihood_BB[i] = log_LBB(T[i], firas_freq.to('s-1'), firas["monopole"], firas["sigma"])
plt.plot(T, likelihood_BB)
plt.xlabel(r'$T$')
plt.ylabel('-2 log(L)')
#plt.yscale('log')
plt.title('Monopole likelihood')
plt.show()



'''
np.random.seed(42)
chain_len = 5000
pos = T_ini*np.random.randn(32, 1)
pos = T_ini + 1e-6*np.random.randn(32, 1)
print(pos.shape)
nwalkers, ndim = pos.shape
sampler = mc.EnsembleSampler(nwalkers, ndim, log_LBB, args = (firas_freq.to('s-1'), firas["monopole"], firas["sigma"]))
sampler.run_mcmc(pos, chain_len, progress = True);

tau = sampler.get_autocorr_time()
print(tau)

samples = sampler.get_chain()
print(samples.shape)
plt.figure()
plt.plot(samples[:, :, 0], "k", alpha = 0.3)
plt.ylabel("T")
plt.xlabel("sample number")
plt.show()


flat_samples = sampler.get_chain(discard = 5*int(tau), thin = int(tau), flat = True)
T_mcmc = np.percentile(flat_samples[:, 0], [16, 50, 84])
q = np.diff(T_mcmc)
print("T = ", T_mcmc[1], "+", q[1], "-", q[0])
'''
np.random.seed(42)
chain_len = 2000
pos = soln_tot.x + 1e-6*np.random.randn(32, 3)
print(pos.shape)
nwalkers, ndim = pos.shape
sampler = mc.EnsembleSampler(nwalkers, ndim, log_Ltot, args = (firas_freq.to('s-1'), firas["monopole"], firas["res"], firas["sigma"]))
sampler.run_mcmc(pos, chain_len, progress = True);

samples = sampler.get_chain()
print(samples.shape)
fig, axes = plt.subplots(sharex=True, nrows=3)
labels=[r'$T_0$', r'$y$', r'$\mu$']
for i in range(3):
    axes[i].plot(samples[50:, :, i], "k", alpha = 0.3)
    axes[i].set_ylabel(labels[i])
    axes[i].yaxis.set_label_coords(-0.1, 0.5)
axes[2].set_xlabel("sample number")
plt.show()
#

flat_samples = sampler.get_chain(discard = 100, thin = 10, flat = True)
from corner import corner
corner(flat_samples, show_titles=True, labels=labels, title_fmt='.6f')
plt.tight_layout()
plt.show()

for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    print(labels[i], ": ", mcmc[1], "+", q[1], "-", q[0])
