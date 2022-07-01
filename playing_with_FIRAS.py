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
names = ["freq", "monopole", "res", "sigma", "gal"]
tmp = pd.read_csv("https://lambda.gsfc.nasa.gov/data/cobe/firas/monopole_spec/firas_monopole_spec_v1.txt", delim_whitespace = True, header = 0, skiprows = 17, names = names)
unit_names = ["cm-1", "MJy/sr", "kJy/sr", "kJy/sr", "kJy/sr"]
units = {names[i]:unit_names[i] for i in range(len(names))}
firas = QTable.from_pandas(tmp, units = units)

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

#set up data to fit to
S = (np.mean(firas_spectra[1].data['spectrum'], axis = 0)*u.MJy/u.sr) #took average over all pixels just to see
#print(S[0:43])
freq = (np.linspace(60, 630, 182)*u.GHz) #there are 182 rows in the dataset

#fit to blackbody function
T_full, pcov_full = curve_fit(func, xdata = freq.si.value, ydata = S.si.value)

bb_full = BlackBody(T_full*u.K) #theoretical blackbody with fitted value
#print(bb_full(freq)[0:43])
print(T_full)


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
#and move on to finding mu following Mather et al (1994)

'''
def func_mu(nu, T, mu):
    #all inputs must be in SI units
    h = const.h.value
    c = const.c.value
    kb = const.k_B.value
    x = h*nu/kb/T
    return (2*h*nu**3/c**2 / (np.exp(x + mu) - 1))
'''

def dBdT(nu, T):
    h = const.h.value
    c = const.c.value
    kb = const.k_B.value
    a = np.exp(h*nu/kb/T)
    return 2*h**2*nu**4/kb/c**2/T**2 * a/(a-1)**2

def func_mu(nu, T, mu):
    T0 = T_simple
    h = const.h.value
    c = const.c.value
    kb = const.k_B.value
    x = h*nu/kb/T
    return -T0*mu/x * dBdT(nu, T)

popt_mu, pcov_mu = curve_fit(func_mu, xdata = firas_freq.si.value, ydata = firas["res"].si.value)
print(popt_mu)

plt.figure()
plt.errorbar(firas_freq, firas["res"], yerr = firas["sigma"], fmt = "k.")
plt.plot(firas_freq, (func_mu(firas_freq.si.value, popt_mu[0], popt_mu[1])*u.J/u.m/u.m/u.sr).to('kJy/sr'), color = 'xkcd:turquoise', ls = "--", label = "fit")
plt.ylim([-100, 100])
plt.xlabel("Frequency (GHz)")
plt.ylabel("Residual intensity (kJy/sr)")
plt.title(r"Temperature + $\mu$")
plt.legend()
plt.show()

#mu is so small that this is visually identical to the plot from before
#interestingly what I find is not the exact same result as Fixsen et all (1996)
#but it is within the error bars
#not including uncertainties within my fit might be part of the difference
#also still need to propagate said errors
#should probably zoom in to show the differences at some point
#should at least be able to see the error bars lol

#to fit y I need to solve the diff eq in Fixsen and then fit the result
#in order to do that I need the derivative of the planck function

def func_y(nu, T, y):
    h = const.h.value
    c = const.c.value
    kb = const.k_B.value
    x = h*c*nu/kb/T
    T0 = T_simple
    return y*(T0*x/np.tanh(x/2) - 4*dBdT(nu, T))

popt_y, pcov_y = curve_fit(func_y, xdata = firas_freq.si.value, ydata = firas["res"].si.value)
print(popt_y)

plt.figure()
plt.errorbar(firas_freq, firas["res"], yerr = firas["sigma"], fmt = "k.")
plt.plot(firas_freq, (func_y(firas_freq.si.value, popt_y[0], popt_y[1])*u.J/u.m/u.m/u.sr).to('kJy/sr'), color = 'xkcd:turquoise', ls = "--", label = "fit")
plt.ylim([-100, 100])
plt.xlabel("Frequency (GHz)")
plt.ylabel("Residual intensity (kJy/sr)")
plt.title("Temperature + y")
plt.legend()
plt.show()


plt.figure()
plt.plot(firas_freq, firas["res"], "k", lw = 1)
plt.plot(firas_freq, (func_mu(firas_freq.si.value, T_simple, 3.3*10**(-4))*u.J/u.m/u.m/u.sr).to('kJy/sr'), color = 'xkcd:turquoise', ls = "--", label = "fit (mu)")
plt.plot(firas_freq, (func_y(firas_freq.si.value, T_simple, 2.5*10**(-5))*u.J/u.m/u.m/u.sr).to('kJy/sr'), color = 'xkcd:orange', ls = "-.", label = "fit (y)")
plt.ylim([-100, 100])
plt.xlabel("Frequency (GHz)")
#plt.ylabel("Residual intensity (kJy/sr)")
plt.title("Reproducing Fig. 5 (Fixsen et al 1996)")
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

popt_mu2, pcov_mu2 = curve_fit(Smu, xdata = firas_2["freq"].value, ydata = firas_2["res"].si.value)
print(popt_mu2) #this is just ones oh nooo
