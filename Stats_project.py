import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelmin
from scipy.stats import chi2

np.random.seed(1)

N_b = 10e5 # Number of background events, used in generation and in fit.
b_tau = 30. # Spoiler.

def generate_data(n_signals = 400):
    ''' 
    Generate a set of values for signal and background. Input arguement sets 
    the number of signal events, and can be varied (default to higgs-like at 
    announcement). 
    
    The background amplitude is fixed to 9e5 events, and is modelled as an exponential, 
    hard coded width. The signal is modelled as a gaussian on top (again, hard 
    coded width and mu).
    '''
    vals = []
    vals += generate_signal( n_signals, 125., 1.5)
    vals += generate_background( N_b, b_tau)
    return vals


def generate_signal(N, mu, sig):
    ''' 
    Generate N values according to a gaussian distribution.
    '''
    return np.random.normal(loc = mu, scale = sig, size = N).tolist()


def generate_background(N, tau):
    ''' 
    Generate N values according to an exp distribution.
    '''
    return np.random.exponential(scale = tau, size = int(N)).tolist()

def exp(x, lamb, A):
    return A*np.exp(-x/lamb)

def signal_gaus(x, mu, sig, signal_amp):
    return signal_amp/(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)


def get_B_expectation(xs, A, lamb):
    ''' 
    Return a set of expectation values for the background distribution for the 
    passed in x values. 
    '''
    return [A*np.exp(-x/lamb) for x in xs]


def get_B_chi(vals, mass_range, nbins, A, lamb):
    ''' 
    Calculates the chi-square value of the no-signal hypothesis (i.e background
    only) for the passed values. Need an expectation - use the analyic form, 
    using the hard coded scale of the exp. That depends on the binning, so pass 
    in as argument. The mass range must also be set - otherwise, its ignored.
    '''
    bin_heights, bin_edges = np.histogram(vals, range = mass_range, bins = nbins)
    half_bin_width = 0.5*(bin_edges[1] - bin_edges[0])
    ys_expected = get_B_expectation(bin_edges + half_bin_width, A, lamb)
    chi = 0

    # Loop over bins - all of them for now. 
    for i in range( len(bin_heights) ):
        chi_nominator = (bin_heights[i] - ys_expected[i])**2
        chi_denominator = ys_expected[i]
        chi += chi_nominator / chi_denominator
    
    return chi # Note B has 2 parameters.


def get_SB_expectation(xs, A, lamb, mu, sig, signal_amp):
    ys = []
    for x in xs:
        ys.append(A*np.exp(-x/lamb) + signal_gaus(x, mu, sig, signal_amp))
    return ys


def get_SB_chi(vals, mass_range, nbins, A, lamb, mu, sig, signal_amp):
    bin_heights, bin_edges = np.histogram(vals, range = mass_range, bins = nbins)
    half_bin_width = 0.5*(bin_edges[1] - bin_edges[0])
    ys_expected = get_SB_expectation(bin_edges + half_bin_width, A, lamb, mu, sig, signal_amp)
    chi = 0
    for i in range( len(bin_heights) ):
        chi_nominator = (bin_heights[i] - ys_expected[i])**2
        chi_denominator = ys_expected[i]
        chi += chi_nominator / chi_denominator
    return chi

#%%
## TASK 1
vals = generate_data()
hist, bin_edges, patches = plt.hist(vals, range = [104, 155], bins = 30)
bincentres = 0.5 * (bin_edges[1:] + bin_edges[:-1])


## TASK 2
x = bincentres[(bincentres <= 120) | (bincentres >= 130)]
y = hist[(bincentres <= 120) | (bincentres >= 130)]
initial_guess = [30, 2e6]
popt, pcov = curve_fit(exp, x, y, p0=initial_guess)

vals = np.array(vals)
mod_vals = vals[(vals <= 120) | (vals >= 130)]
likelihood_lambda = np.mean(mod_vals)
likelihood_A = 60441.47454

B = len(mod_vals)/likelihood_lambda**2 - 2/likelihood_lambda**3 * np.sum(mod_vals)
lambda_unc = np.sqrt(1/(-B))


# TASK 3
lambda_test = np.linspace(25, 35, num=1000)
chi_lambda = get_B_chi(vals, (104, 155), 30, popt[1], lambda_test)
min_chi_lambda = argrelmin(chi_lambda)

A_test = np.linspace(50000, 70000, num=1000)
chi_A = get_B_chi(vals, (104, 155), 30, A_test, popt[0])
min_chi_A = argrelmin(chi_A)

min_chi_B = get_B_chi(vals, (104, 155), 30, A_test[min_chi_A], lambda_test[min_chi_lambda])


# TASK 4
min_chi_SB = get_SB_chi(vals, (104, 155), 30, A_test[min_chi_A], lambda_test[min_chi_lambda], 125, 1.5, 700)

signal_amp_test = np.linspace(0, 1400, num=1000)
chi_signal_amp = get_SB_chi(vals, (104, 155), 30, popt[1], popt[0], 125, 1.5, signal_amp_test)
pvals = 1 - chi2.cdf(chi_signal_amp, 27)
plt.figure()
plt.plot(signal_amp_test, pvals, label="p-values")
plt.axhline(y=0.05, color="red", linestyle="--", label="p-value = 0.05")
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xlabel("Signal amplitude", fontsize=14)
plt.ylabel("p-value", fontsize=14)
plt.grid()
plt.legend(fontsize=13)
plt.show()
plt.close()

mu_test = np.linspace(50, 200, num=1000)
chi_mu = get_SB_chi(vals, (0, 250), 425, popt[1], popt[0], mu_test, 1.5, 700)
plt.figure()
plt.plot(mu_test, chi_mu, label="Chi-squared values")
plt.axvline(x=125, color="red", linestyle="--", label="μ = 125 GeV")
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xlabel("Signal mean (μ)/GeV", fontsize=14)
plt.ylabel("Chi-squared value", fontsize=14)
plt.grid()
plt.legend(fontsize=13)
plt.show()

plt.figure()
plt.clf()
plt.errorbar(bincentres, hist, yerr=np.sqrt(hist), fmt='.', capsize=2, label="Data")
plt.plot(bincentres, exp(bincentres, popt[0], popt[1]), label="Fitted curve")
plt.plot(bincentres, exp(bincentres, likelihood_lambda, likelihood_A), label="Max likelihood")
plt.plot(bincentres, exp(bincentres, lambda_test[min_chi_lambda], A_test[min_chi_A]), label="Min chi-squared")
plt.xlabel("Mass/GeV")
plt.ylabel("Number of entries")
plt.grid()
plt.legend()
plt.show()
plt.close()
