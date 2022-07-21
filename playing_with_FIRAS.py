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
plt.rc('figure', figsize = (7, 7/1.25))
plt.rc('font', size = 14)

#get the FIRAS data from their paper:
#names = ["freq", "monopole", "res", "sigma", "gal"]
#tmp = pd.read_csv("https://lambda.gsfc.nasa.gov/data/cobe/firas/monopole_spec/firas_monopole_spec_v1.txt", delim_whitespace = True, header = 0, skiprows = 17, names = names)
#unit_names = ["cm-1", "MJy/sr", "kJy/sr", "kJy/sr", "kJy/sr"]
#units = {names[i]:unit_names[i] for i in range(len(names))}
#firas = QTable.from_pandas(tmp, units = units)
freq = [2.27, 2.72, 3.18, 3.64, 4.08, 4.54, 4.99, 5.45, 5.90, 6.35, 6.81, 7.26, 7.71, 8.17, 8.62, 9.08, 9.53, 9.98, 10.44, 10.89, 11.34, 11.80, 12.25, 12.71, 13.16, 13.61, 14.07, 14.52, 14.97, 15.43, 15.88, 16.34, 16.79, 17.24, 17.70, 18.15, 18.61, 19.06, 19.51, 19.97, 20.42, 20.87, 21.33]/u.cm
monopole = [200.723, 249.508, 293.024, 327.770, 354.081, 372.079, 381.493, 383.478, 378.901, 368.833, 354.063, 336.278, 316.076, 293.924, 271.432, 248.239, 225.940, 204.327, 183.262, 163.830, 145.750, 128.835, 113.568, 99.451, 87.036, 75.876, 65.766, 57.008, 49.223, 42.267, 36.352, 31.062, 26.580, 22.644, 19.255, 16.391, 13.811, 11.716, 9.921, 8.364, 7.087, 5.801, 4.523]*u.MJy/u.sr
res = [5, 9, 15, 4, 19, -30, -30, -10, 32, 4, -2, 13, -22, 8, 8, -21, 9, 12, 11, -29, -46, 58, 6, -6, 6, -17, 6, 26, -12, -19, 8, 7, 14, -33, 6, 26, -26, -6, 8, 26, 57, -116, -432]*u.kJy/u.sr
sigma = [14, 19, 25, 23, 22, 21, 18, 18, 16, 14, 13, 12, 11, 10, 11, 12, 14, 16, 18, 22, 22, 23, 23, 23, 22, 21, 20, 19, 19, 19, 21, 23, 26, 28, 30, 32, 33, 35, 41, 55, 88, 155, 282]*u.kJy/u.sr
gal = [4, 3, -1, -1, 3, 6, 8, 8, 10, 10, 12, 20, 25, 30, 36, 41, 46, 57, 65, 73, 93, 98, 105, 121, 135, 147, 160, 178, 199, 221, 227, 250, 275, 295, 312, 336, 363, 405, 421, 435, 477, 519, 573]*u.kJy/u.sr
firas = QTable([freq, monopole, res, sigma, gal], names = ("freq", "monopole", "res", "sigma", "gal"))

# COBE/FIRAS CMB monopole spectrum
# Reference1 = Table 4 of Fixsen et al. 1996 ApJ 473, 576.
# Reference2 = Fixsen & Mather 2002 ApJ 581, 817.
# Column 1 = frequency from Table 4 of Fixsen et al., units = cm^-1
# Column 2 = FIRAS monopole spectrum computed as the sum
#             of a 2.725 K BB spectrum and the
#             residual in column 3, units = MJy/sr
# Column 3 = residual monopole spectrum from Table 4 of Fixsen et al.,
#             units = kJy/sr
# Column 4 = spectrum uncertainty (1-sigma) from Table 4 of Fixsen et al.,
#             units = kJy/sr
# Column 5 = modeled Galaxy spectrum at the Galactic poles from Table 4 of
#             Fixsen et al., units = kJy/sr

#raw data:
firas_spectra = fits.open('FIRAS_DESTRIPED_SKY_SPECTRA_LOWF.FITS')

### fit a blackbody spectrum to data

#blackbody function:
#I get errors trying to do this with astropy's built-in
#because scipy expects dimensionless quantities
def func(nu, T):
    #nu and T must both be in SI units
    h = const.h.value
    c = const.c.value
    kb = const.k_B.value
    return (2*h*nu**3/c**2 / (np.exp(h*nu/kb/T) - 1))

'''
#set up data to fit to
S = (np.mean(firas_spectra[1].data['spectrum'], axis = 0)*u.MJy/u.sr) #took average over all pixels just to see
#print(S[0:43])
freq = (np.linspace(60, 630, 182)*u.GHz) #there are 182 rows in the dataset

#fit to blackbody function
T_full, pcov_full = curve_fit(func, xdata = freq.si.value, ydata = S.si.value)

bb_full = BlackBody(T_full*u.K) #theoretical blackbody with fitted value
#print(bb_full(freq)[0:43])
print(T_full)
'''


### also try with data from Fixsen (1996):
firas_freq = firas["freq"].to('GHz', equivalencies = u.spectral())
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

'''
plt.figure()
for i in range(100):
    #iterates over all pixels: there are 6067 total but don't plot that, it'll kill your computer
    plt.plot(freq, firas_spectra[1].data['spectrum'][i], 'k.', alpha=0.5)
plt.plot(freq, bb_full(freq).to('MJy/sr'), color = 'xkcd:turquoise', ls = "--", label = "fit 1")
plt.plot(firas_freq, bb_simple(firas_freq).to('MJy/sr'), color = 'xkcd:orange', ls = "--", label = "fit 2")
plt.xlim(xmax = freq[43])
plt.ylabel("Intensity (MJy/sr)")
plt.xlabel("Frequency (GHz)")
plt.legend()
plt.show()
'''

#ok neither of these are decent fits at all and I have no idea why
#thought initially the problem was that I was averaging over the whole sky differently
#but the Fixsen data wouldn't have that problem and there's still a giant discrepancy
#between the "fitted" spectrum and the actual spectrum

#update: the Fixsen one is now.....fixed. so the problem is def the averaging
#result agrees with published value, although I haven't propagated the uncertainties yet
#update: don't know what it is but it's sure not the averaging, at least not entirely

#update: something's weird with the raw data, so I'm gonna ignore it for now
#and move on to finding mu following Fiksen et al (1996)

#equation 4 in Fiksen et al 1996 is in units of W/m/sr
#intensities are given in kJy/sr, which can be converted to W/m^2/sr
#so somewhere I'm missing a factor of 1/m
#technically the intensity is p dS/dp
#so is the spectral distortion given in 1/m?
#no, it should be dimensionless
#edit: found it, the thing that should be in kJy/sr is T dBdT, not S

def dBdT(nu):
    #we want nu in s-1
    #all constants in SI units
    T = T_simple*u.K
    x = const.h*nu/const.k_B/T
    return 2 * const.h**2 * nu**4 / (const.k_B * const.c**2 * T**2) * np.exp(x) / np.expm1(x)**2

def func_mu(nu, mu):
    #nu in GHz (but without astropy units or scipy freaks out), T in K
    #all constants in SI units
    nu = (nu*u.GHz).to('s-1')
    T = T_simple*u.K
    x = const.h*nu/const.k_B/T
    return (-T*mu/x * dBdT(nu)).to('kJy').value


popt_mu, pcov_mu = curve_fit(func_mu, xdata = firas_freq.value, ydata = firas["res"])
print(popt_mu)

plt.figure()
#plt.errorbar(firas_freq, firas["res"], yerr = firas["sigma"], fmt = "k-", lw = 1)
plt.plot(firas_freq, firas["res"], "k", lw = 1)
plt.plot(firas_freq, (func_mu(firas_freq.value, popt_mu[0]))*u.kJy/u.sr, color = 'xkcd:turquoise', ls = "--", label = "fit")
plt.ylim([-100, 100])
plt.xlabel("Frequency (GHz)")
plt.ylabel("Residual intensity (kJy/sr)")
plt.title(r"Temperature + $\mu$")
plt.tight_layout()
plt.legend()
plt.show()

#mu is so small that this is visually identical to the plot from before
#interestingly what I find is not the exact same result as Fixsen et all (1996)
#but it is within the error bars
#not including uncertainties within my fit might be part of the difference
#also still need to propagate said errors
#should probably zoom in to show the differences at some point
#should at least be able to see the error bars lol
#edit: fit the residuals instead and this is no longer true

#to fit y I need to solve the diff eq in Fixsen and then fit the result
#in order to do that I need the derivative of the planck function
#edit: this is not true

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
#plt.errorbar(firas_freq, firas["res"], yerr = firas["sigma"], fmt = "k.")
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
plt.plot(firas_freq, (func_mu(firas_freq.value, -9*10**(-5)))*u.kJy/u.sr, color = 'xkcd:turquoise', ls = "--", label = "fit (mu)")
plt.plot(firas_freq, (func_y(firas_freq.value, 15*10**(-6))*u.kJy/u.sr), color = 'xkcd:orange', ls = "-.", label = "fit (y)")
plt.ylim([-100, 100])
plt.xlabel("Frequency (GHz)")
plt.ylabel("Residual intensity (kJy/sr)")
plt.title("Reproducing Fig. 5 (Fixsen et al 1996)")
plt.tight_layout()
plt.legend()
plt.show()

#this is complaining a lot and I feel like it's probably either that I have a typo somewhere
#or it's having overflow issues or something
#cause the temperature it's coming up with is just 1 K
#which makes me think overflow because it doesn't seem like a random typo kind of error to get
#or you know something else I'm not thinking of
#I might just come back to this tomorrow

#update: found the problem, I didn't write down the right expression for S
#forgot the differential equation was for the linearized distortion
#so you can't actually solve it and get the full expression
#which is very annoying

#update: got it!!! I need to fit to the residuals instead of the monopole!
#for both mu and y actually

#hmmm this gets me very incorrect values but no overflow errors
#not entirely sure if that's an improvement
#but I'm pretty sure I'm on the right track

### gonna try the data from mather et al 1996
'''
freq = np.array([2.27, 2.83, 3.40, 3.96, 4.53, 5.10, 5.66, 6.23, 6.80, 7.36, 7.93, 8.49, 9.06, 9.63, 10.19, 10.76, 11.33, 11.89, 12.46, 13.03, 13.59, 14.16, 14.72, 15.29, 15.86, 16.42, 16.99, 17.56, 18.12, 18.69, 19.26, 19.82, 20.39, 20.95])/u.cm
res = np.array([18, 19, 34, 10, 2, -24, 4, -5, -6, -4, -2, 6, -2, 7, 4, -28, -9, 17, 2, 16, 4, 4, 15, -4, 8, 17, 6, -1, -12, 0, -15, -8, 25, -88])*10**(-9)*u.erg/u.cm/u.cm/u.s/u.sr*u.cm
sigma = np.array([9, 14, 16, 14, 14, 12, 12, 9, 9, 8, 7, 7, 8, 10, 11, 13, 15, 14, 14, 13, 12, 11, 11, 12, 13, 15, 17, 18, 19, 20, 20, 25, 40, 78])*10**(-9)*u.erg/u.cm/u.cm/u.s/u.sr*u.cm
gal = np.array([1, 1, 2, 2, 2, 4, 4, 7, 9, 15, 18, 18, 23, 28, 35, 42, 55, 59, 65, 74, 85, 97, 108, 129, 136, 163, 166, 184, 202, 222, 247, 261, 282, 306])*10**(-9)*u.erg/u.cm/u.cm/u.s/u.sr*u.cm

firas_2 = QTable([freq, res, sigma, gal], names = ("freq", "res", "sigma", "gal"))

def dBdT2(nu, T):
    h = const.h.value
    c = const.c.value
    kb = const.k_B.value
    x = h*c*nu/kb/T
    a = np.exp(h*c*nu/kb/T)
    return 2*h**2*nu**4*c**2/kb/T**2 * a/(a-1)**2
def Smu(nu, T, mu):
    T0 = 2.725
    h = const.h.value
    c = const.c.value
    kb = const.k_B.value
    x = h*c*nu/kb/T
    return -T0*mu/x *dBdT(nu, T)

popt_mu2, pcov_mu2 = curve_fit(Smu, xdata = firas_2["freq"].value, ydata = firas_2["res"].value)
print(popt_mu2) #this is just ones oh nooo
'''


### now that y works, at least, we can try fitting using mcmc
#first: likelihood function
def log_Ly(theta, nu, Sres, Serr):
    #nu is an array of frequencies in s-1
    #Sres and Serr are the residual and error arrays in kJy/sr
    #T, y = theta
    T = theta
    print(T)
    y = -1e-6
    x = const.h*nu/const.k_B/(T*u.K)
    print(np.exp(x))
    model = (y*T*u.K * (x * (np.exp(x)+1) / np.expm1(x) - 4) * dBdT(nu)).to('kJy')/u.sr
    #print(model)
    sigma2 = Serr**2 + model**2
    return -0.5 * np.sum((Sres-model)**2/sigma2)

def log_Lmu(theta, nu, Sres, Serr):
    T, mu = theta
    x = const.h*nu/const.k_B/(T*u.K)
    model = (-T*u.K*mu/x * dBdT(nu)).to('kJy')/u.sr
    sigma2 = Serr**2 + model**2
    return -0.5 * np.sum((Sres-model)**2/sigma2)


plt.figure()
plt.plot(firas_freq.to('s-1'), firas["res"], lw = 1, color = 'k', label = r"S($\nu$)")
x = const.h*firas_freq.to('s-1')/const.k_B/(T_simple*u.K)
y = -1e-6
mu = -1e-5
model_y = (y*T_simple*u.K * (x * (np.exp(x)+1) / np.expm1(x) - 4) * dBdT(firas_freq.to('s-1'))).to('kJy')/u.sr
model_mu = (-T_simple*u.K*mu/x * dBdT(firas_freq.to('s-1'))).to('kJy')/u.sr
plt.plot(firas_freq.to('s-1'), np.sqrt(firas["sigma"]**2 + model_y**2), color = 'gray', label = r"$\sigma$")
plt.plot(firas_freq.to('s-1'), model_y, lw = 1, ls = '--', color = "xkcd:orange", label = "f(T, y)")
#plt.plot(firas_freq.to('s-1'), model_mu, lw = 1, ls = '--', color = "xkcd:turquoise")
plt.ylim([-100, 100])
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#maximum likelihood:
from scipy.optimize import minimize

nll_y = lambda *args: -log_Ly(*args)
nll_mu = lambda *args: -log_Lmu(*args)
T_ini = 2.728
y_ini = -1e-6
mu_ini = -1e-5
initial_y = np.array([T_ini, y_ini])
initial_mu = np.array([T_ini, mu_ini])
initial = np.array([T_ini])
print("Initial:")
print(T_ini)
print(y_ini)
print(mu_ini)

soln_y = minimize(nll_y, initial, args = (firas_freq.to('s-1'), firas["res"], firas["sigma"]))
soln_mu = minimize(nll_mu, initial_mu, args = (firas_freq.to('s-1'), firas["res"], firas["sigma"]))
T_ml_y = soln_y.x
T_ml_mu, mu_ml = soln_mu.x

print("Maximum likelihood estimates:")
print(T_ml_y, ",  ", T_ml_mu)
#print(y_ml)
print(mu_ml)
#y gives overflow warnings and spits out whatever I put in
#mu doesn't give warnings but spits out nonsense

#print(y_ml)
#print(mu_ml)

'''
np.random.seed(42)
pos = T_ini*np.random.randn(32, 1)
#pos = soln_y.x + [1e-4, 1e-7]*np.random.randn(32, 2)
print(pos.shape)
nwalkers, ndim = pos.shape
sampler = mc.EnsembleSampler(nwalkers, ndim, log_Ly, args = (firas_freq.to('s-1'), firas["res"], firas["sigma"]))
sampler.run_mcmc(pos, 5000, progress = True);
'''
