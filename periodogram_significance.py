"""


@author: ahana
"""

import xarray as xr
import matplotlib.pyplot as plt
import cmasher as cmr
import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import linregress
import matplotlib.dates as mdates
from astropy.timeseries import LombScargle
from datetime import datetime

a = xr.open_dataset('/home/ahana/Documents/rdr_filename/rdr45_tke_shr_wnd_2017-22_95%(-5,5)filt.nc').sel(level=slice(0,3))
b = xr.open_dataset('/home/ahana/Documents/rdr_filename/rdr18_tke_shr_wnd_2017-22_95%(-5,5)filt.nc').sel(level=slice(3,12))
#%%

c = a.tke.mean('level')#.resample(time='1W').mean('time')
c = c[~np.isnan(c)]
d = b.tke.mean('level')#.resample(time='1W').mean('time')
d = d[~np.isnan(d)]

dnum1 = mdates.date2num(c.time)
# xx_e = dnum * 604800 / 3600
xx_e1 = dnum1 *86400      # 60*60*24   [86400 - number time in c]
y1 = c[~np.isnan(c)].values

# Lomb-Scargle periodogram analysis
ls1 = LombScargle(xx_e1, y1)

ww1, pow1 = ls1.autopower(normalization='standard', nyquist_factor=1)
ww1 = 2 * np.pi * ww1 # in rad/sec
ls1.false_alarm_probability(pow1.max())
# Convert frequencies to periods
periods_lomb1 = 1 / (ww1*(1/(2*np.pi))) #------> in seconds
pl1 = periods_lomb1/86400

sx1 = pl1[[list(pow1).index(x) for x in sorted(pow1, reverse=True)][:10]]
sy1 = pow1[[list(pow1).index(x) for x in sorted(pow1, reverse=True)][:10]]

fap1 = ls1.false_alarm_probability(sy1)
# pow1 = pow1[1:]
# pl1 = pl1[1:]
sy1[fap1<0.1]
sx1[fap1<0.1]
#------------------------------------------------------------------------------------------------------------------
dnum2 = mdates.date2num(d.time)
# xx_e = dnum * 604800 / 3600
xx_e2 = dnum2 *86400      # 60*60*24*7   [304 - number time in c]
y2 = d.values

# Lomb-Scargle periodogram analysis
# ww2, pow2 = LombScargle(xx_e2, y2).autopower(nyquist_factor=1, normalization='psd')

ls2 = LombScargle(xx_e2, y2)

ww2, pow2 = ls2.autopower(normalization='standard', nyquist_factor=1)
ww2 = 2 * np.pi * ww2 # in rad/sec

# Convert frequencies to periods
periods_lomb2 = 1 / (ww2*(1/(2*np.pi))) #------> in seconds
pl2 = periods_lomb2/86400

sx2 = pl2[[list(pow2).index(x) for x in sorted(pow2, reverse=True)][:20]]
sy2 = pow2[[list(pow2).index(x) for x in sorted(pow2, reverse=True)][:20]]

fap2 = ls2.false_alarm_probability(sy2)
# pow2 = pow2[2:]
# pl2 = pl2[2:]
sy2[fap2<0.05]       #----> pow2 values with false alarm probability values less than 0.05
sx2[fap2<0.05]       #----> ww2 values with false alarm probability values less than 0.05

#%%

fig, (ax, ax1) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
ax.plot(pl1, pow1, c= 'k')
# plt.xlabel('Period (hours)')
# plt.ylabel('Power')
ax.scatter(sx1[fap1<=0.05], sy1[fap1<=0.05], c = 'r', s = 75, marker ='*', label = 'significant peaks')
# ax.set_xlabel('Period', fontsize = 14)
# ax.set(xlim=(0, 750), xticks = [30,60, 180, 365, 730], xticklabels = ['1m', '2m', '6m', '1y', '2y'])
ax.set_ylabel(r'$PSD\ (m^4\ s^{-6})$', fontsize = 14)
ax.set_title('Lower Troposphere')
ax.set(xlim=(-10, 1460), ylim = (0,0.08), xticks = [31,59, 120, 181, 365, 730, 1460], 
        yticks = np.arange(0,0.081,0.01), xticklabels = [],)
        # yticklabels = np.round(np.arange(0,0.08,0.002)*10000,2))
ax.legend(fontsize = 15)
ax.tick_params(axis = 'both', labelsize=14)

ax1.plot(pl2[2:], pow2[2:], c= 'k')
ax1.scatter(sx2[fap2<0.05], sy2[fap2<0.05], c = 'r',  s = 75, marker ='*', label = 'significant peaks')
ax1.set_xlabel('Period', fontsize = 14)
ax1.set_ylabel(r'$PSD\ (m^4\ s^{-6})$', fontsize = 14)
ax1.set_title('Middle Troposphere')
ax1.set(xlim=(-10, 1460), ylim = (0,0.045), xticks = [59, 120, 181, 365, 730, 1460], 
        yticks = np.arange(0,0.046,0.005), xticklabels = ['2m', '4m', '6m', '1y', '2y', '4y'],)
        # yticklabels = np.rounnp.arange(0,0.0004,0.0001)*10000,2)
# ax1.set_title('Normalisation - standard')
ax1.tick_params(axis = 'both', labelsize=14)
ax1.legend(fontsize = 15)
fig.tight_layout()
fig.savefig('/home/ahana/Documents/pp4/fig/Periodogram_standard1.jpg', dpi = 300)


#%%%
"""
da3 = a.tke.sel(level = slice(0, 3)).mean('level')
da6 = b.tke.sel(level = slice(3, 6)).mean('level')
da9 = b.tke.sel(level = slice(6, 9)).mean('level')
da12 = b.tke.sel(level = slice(9, 12)).mean('level')
da = [da3, da6, da9, da12]

#%%

fig = plt.figure(figsize=(11, 8))
for k in range(len(da)):
    
    c = da[k][~np.isnan(da[k])]

    dnum1 = mdates.date2num(c.time)
    # xx_e = dnum * 604800 / 3600
    xx_e1 = dnum1 *86400      # 60*60*24   [86400 - number time in c]
    y1 = c[~np.isnan(c)].values
    
    # Lomb-Scargle periodogram analysis
    ls1 = LombScargle(xx_e1, y1)
    
    ww1, pow1 = ls1.autopower(normalization='standard', nyquist_factor=1)
    ww1 = 2 * np.pi * ww1 # in rad/sec
    ls1.false_alarm_probability(pow1.max())
    # Convert frequencies to periods
    periods_lomb1 = 1 / (ww1*(1/(2*np.pi))) #------> in seconds
    pl1 = periods_lomb1/86400
    
    sx1 = pl1[[list(pow1).index(x) for x in sorted(pow1, reverse=True)][:10]]
    sy1 = pow1[[list(pow1).index(x) for x in sorted(pow1, reverse=True)][:10]]
    
    fap1 = ls1.false_alarm_probability(sy1)
    # pow1 = pow1[1:]
    # pl1 = pl1[1:]
    sy1[fap1<0.1]
    sx1[fap1<0.1]
    ax = fig.add_subplot(4,1,k+1)
    ax.plot(pl1, pow1, c= 'k')
    ax.scatter(sx1[fap1<0.1], sy1[fap1<0.1], c = 'r', s = 75, marker ='*', label = 'significant peaks')
    ax.set(xlim=(-10, 1460), xticks = [59, 120, 181, 365, 730, 1095, 1460, 1825], 
           xticklabels = ['2m', '4m', '6m', '1y', '2y', '3y', '4y', '5y'],)
    ax.tick_params(axis = 'both', labelsize=14)
    ax.legend(fontsize = 15)
fig.tight_layout()

"""