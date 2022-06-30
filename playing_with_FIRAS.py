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

def func_mu(nu, T, mu):
    #all inputs must be in SI units
    h = const.h.value
    c = const.c.value
    kb = const.k_B.value
    x = h*nu/kb/T
    return (2*h*nu**3/c**2 / (np.exp(x + mu) - 1))

popt_mu, pcov_mu = curve_fit(func_mu, xdata = firas_freq.si.value, ydata = firas["monopole"].si.value)
print(popt_mu)

plt.figure()
plt.errorbar(firas_freq, firas["monopole"], yerr = firas["sigma"], fmt = "k.")
plt.plot(firas_freq, (func_mu(firas_freq.si.value, popt_mu[0], popt_mu[1])*u.J/u.m/u.m/u.sr).to('MJy/sr'), color = 'xkcd:turquoise', ls = "--", label = "fit")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Intensity (MJy/sr)")
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

def dBdT(nu, T):
    h = const.h.value
    c = const.c.value
    kb = const.k_B.value
    a = h*nu/kb
    return 2*h*nu**3/c**2 * a*np.exp(a/T)/(T*(np.exp(a/T) - 1))**2

def func_y(nu, T, y):
    h = const.h.value
    c = const.c.value
    kb = const.k_B.value
    x = h*nu/kb/T
    T0 = T_simple
    return y*(T0*x/np.tanh(x/2) - 4*dBdT(nu, T))

popt_y, pcov_y = curve_fit(func_y, xdata = firas_freq.si.value, ydata = firas["monopole"].si.value)
print(popt_y)

plt.figure()
plt.errorbar(firas_freq, firas["monopole"], yerr = firas["sigma"], fmt = "k.")
plt.plot(firas_freq, (func_y(firas_freq.si.value, popt_y[0], popt_y[1])*u.J/u.m/u.m/u.sr).to('MJy/sr'), color = 'xkcd:turquoise', ls = "--", label = "fit")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Intensity (MJy/sr)")
plt.title("Temperature + y")
plt.legend()
plt.show()

#this is complaining a lot and I feel like it's probably either that I have a typo somewhere
#or it's having overflow issues or something
#cause the temperature it's coming up with is just 1 K
#which makes me think overflow because it doesn't seem like a random typo kind of error to get
#or you know something else I'm not thinking of
#I feel guilty because it doesn't feel like I got a lot of work done today but I'm really tired
#so I might just come back to this tomorrow
