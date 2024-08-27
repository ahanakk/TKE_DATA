""""

@author: ahana
"""
#%%
from scipy import stats
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cmasher as cmr
import metpy.calc as mc
from metpy.units import units
from season import jjas, djf, mam, ON
from scipy.stats import linregress, t
from matplotlib import gridspec
from matplotlib.lines import Line2D
import pandas as pd

a45 = xr.open_dataset('/home/ahana/Documents/rdr_filename/rdr45_tke_shr_wnd_2017-22_95%(-5,5)filt.nc').sel(level=slice(0,3))
a18 = xr.open_dataset('/home/ahana/Documents/rdr_filename/rdr18_tke_shr_wnd_2017-22_95%(-5,5)filt.nc').sel(level=slice(3,12))
#%%
x = a45.wnd.values[~np.isnan(a45.wnd.values)]
y = np.log10(a45.tke).values[~np.isnan(np.log10(a45.tke).values)]


skew = stats.skew(np.log10(a45.tke).values, nan_policy='omit')

xmin = 0
xmax = 30

ymin = -8.5
ymax = -0.3

X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])

values = np.vstack([x, y])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)
kurt = stats.kurtosis(Z.T)

x1 = a18.wnd.values[~np.isnan(a18.wnd.values)]
y1 = np.log10(a18.tke).values[~np.isnan(np.log10(a18.tke).values)]

xmin1 = 0
xmax1 = 30

ymin1 = -8.5
ymax1 = -0.3

X1,  Y1 = np.mgrid[xmin1 : xmax1 : 100j, ymin1 : ymax1 : 100j]
positions1 = np.vstack([X1.ravel(), Y1.ravel()])

values1 = np.vstack([x1, y1])
kernel1 = stats.gaussian_kde(values1)
Z1 = np.reshape(kernel1(positions1).T, X1.shape)
kurt1 = stats.kurtosis(Z1.T)
skew1 = stats.skew(np.log10(a18.tke).values, nan_policy='omit')

x1 = a18.wnd.values[~np.isnan(a18.wnd.values)]
y1 = np.log10(a18.tke).values[~np.isnan(np.log10(a18.tke).values)]

xmin1 = 0
xmax1 = 30
X18 = np.linspace(xmin1, xmax1, 100)
kurt18 = np.zeros(100)
skew18 = np.zeros(100)
for i in range(len(X18)-1):
    tk = a18.where((a18.wnd >= X18[i]) & (a18.wnd < X18[i+1])).tke.values
    tk = tk[~np.isnan(tk)]
    kurt18[i] = stats.kurtosis(tk, nan_policy='omit')
    skew18[i] = stats.skew(tk, nan_policy='omit')
    
x2 = a45.wnd.values[~np.isnan(a45.wnd.values)]
y2 = np.log10(a45.tke).values[~np.isnan(np.log10(a45.tke).values)]


xmin2 = 0
xmax2 = 30
X45 = np.linspace(xmin2, xmax2, 100)
kurt45 = np.zeros(100)
skew45 = np.zeros(100)
for i in range(len(X45)-1):
    tk = a45.where((a45.wnd >= X45[i]) & (a45.wnd < X45[i+1])).tke.values
    tk = tk[~np.isnan(tk)]
    kurt45[i] = stats.kurtosis(tk, nan_policy='omit')
    skew45[i] = stats.skew(tk, nan_policy='omit')
    

#%%

fig = plt.figure(figsize = (8, 12))
 
# to change size of subplot's
# set height of each subplot as 8
fig.set_figheight(8)
 
# set width of each subplot as 8
fig.set_figwidth(8)
 
# create grid for different subplots
spec = gridspec.GridSpec(ncols=1, nrows=3,
                         # wspace=0.02,  width_ratios=[2, 2,2],
                         hspace=0.1, height_ratios=[4, 1, 1])
ax = fig.add_subplot(spec[0])
sm = ax.contourf(Z.T, cmap='Purples', vmin = 0, vmax = 0.12, levels = 10, 
                 extent=[xmin, xmax, ymin, ymax])
sm1 = ax.contour(Z1.T, cmap='cmr.ember_r', vmin = 0, vmax = 0.12, levels = 10,
                 extent=[xmin1, xmax1, ymin1, ymax1])

# ax1.plot(X1, kurt1, 'k.', markersize=2)

ax.set(xlim = (0,25), ylim  =(-5,-1))
# plt.colorbar(sm)
plt.clabel(sm1, inline=True, fontsize=8, colors='k', fmt='%1.2f')
# ax.set_xlabel(r'$Wind\ Speed\ (m\ s^{-1})$', fontsize =14)
ax.set(xticklabels=[])
ax.set_ylabel(r'$log(\varepsilon)\ (m^2\ s^{-3})$', fontsize =14)
ax.tick_params(axis = 'both', labelsize = 14)
cbar_ax = fig.add_axes([0.9, 0.56, 0.02, 0.4])  # Adjust position and size of the colorbar
sm = plt.cm.ScalarMappable(cmap="Purples", norm=plt.Normalize(vmin=0, vmax=0.12))
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax, ticks=np.linspace(0, 0.12, 10), boundaries=np.linspace(0, 0.12, 10), 
             pad = 0.02)


ax1 = fig.add_subplot(spec[1])

ax1.plot(X18, pd.Series(skew18).rolling(9).mean(), 'k', )
ax1.plot(X45, pd.Series(skew45).rolling(9).mean(), 'r')
ax1.tick_params(axis = 'y', labelsize = 14)

ax1.set(xlim = (0, 30), ylim = (0,25), xticklabels = [], yticks = [0, 10, 20])
ax1.set_ylabel('Skewness', fontsize=14)
 
ax2 = fig.add_subplot(spec[2])

ax2.plot(X18, pd.Series(kurt18).rolling(9).mean(), 'k')
ax2.plot(X45, pd.Series(kurt45).rolling(9).mean(), 'r')
ax2.set(xlim = (0, 30), ylim= (0, 750), yticks = np.arange(0,1000,250), 
        yticklabels = (np.arange(0,1000,250)/10).astype('int'))
ax2.tick_params(axis = 'both', labelsize = 14)
ax2.set_xlabel(r'$Wind\ Speed\ (m\ s^{-1})$', fontsize= 16)
ax2.set_ylabel('Kurtosis', fontsize=14)

c = ['k', 'r',]
l1 = [Line2D([0], [0], color=c, linewidth=2, linestyle='-') for c in c[:2]]
labels = [ 'Mid-Troposphere', 'Lower Troposphere',]
fig.legend(l1, labels, loc='center', fontsize =11, bbox_to_anchor=(0.63, 0.38), ncol=2)

fig.text(0.13, 0.95, '(a)', fontsize = 16)
fig.text(0.13, 0.35, '(b)', fontsize = 16)
fig.text(0.13, 0.19, '(c)', fontsize = 16)
fig.text(0.1, 0.23, r'$\times\ 10$')
fig.subplots_adjust(top = 0.99, right = 0.89, bottom =0.08, left =0.11,)
fig.savefig('/home/ahana/Documents/pp4/fig/paper/tke_wind_gkde_kur_skew.jpg', dpi = 300)

#%%


xx = a45.duz.values[~np.isnan(a45.duz.values)]
y = np.log10(a45.tke).values[~np.isnan(np.log10(a45.tke).values)]

xmin = -0.18
xmax = 0.16

ymin = -8.3
ymax = -0.4

X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
positions = np.vstack([X.ravel(), Y.ravel()])

values = np.vstack([xx, y])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)

xx1 = a18.duz.values[~np.isnan(a18.duz.values)]
y1 = np.log10(a18.tke).values[~np.isnan(np.log10(a18.tke).values)]

xmin1 = -0.18
xmax1 =0.16

ymin1 = -8.3
ymax1 = -0.4

X1,  Y1 = np.mgrid[xmin1 : xmax1 : 200j, ymin1 : ymax1 : 200j]
positions1 = np.vstack([X1.ravel(), Y1.ravel()])

values1 = np.vstack([xx1, y1])
kernel1 = stats.gaussian_kde(values1)
Z1 = np.reshape(kernel1(positions1).T, X1.shape)
x1 = a18.duz.values[~np.isnan(a18.duz.values)]
y1 = np.log10(a18.tke).values[~np.isnan(np.log10(a18.tke).values)]

xmin1 = -0.18
xmax1 = 0.16
X18 = np.linspace(xmin1, xmax1, 200)
kurt18 = np.zeros(200)
skew18 = np.zeros(200)
for i in range(len(X18)-1):
    tk = a18.where((a18.duz >= X18[i]) & (a18.duz < X18[i+1])).tke.values
    tk = tk[~np.isnan(tk)]
    kurt18[i] = stats.kurtosis(tk, nan_policy='omit')
    skew18[i] = stats.skew(tk, nan_policy='omit')
    
x2 = a45.duz.values[~np.isnan(a45.duz.values)]
y2 = np.log10(a45.tke).values[~np.isnan(np.log10(a45.tke).values)]


xmin2 = -0.18
xmax2 = 0.16
X45 = np.linspace(xmin2, xmax2, 200)
kurt45 = np.zeros(200)
skew45 = np.zeros(200)
for i in range(len(X45)-1):
    tk = a45.where((a45.duz >= X45[i]) & (a45.duz < X45[i+1])).tke.values
    tk = tk[~np.isnan(tk)]
    kurt45[i] = stats.kurtosis(tk, nan_policy='omit')
    skew45[i] = stats.skew(tk, nan_policy='omit')
    

    
#%%


fig = plt.figure(figsize = (8, 12))
 
# to change size of subplot's
# set height of each subplot as 8
fig.set_figheight(8)
 
# set width of each subplot as 8
fig.set_figwidth(8)
 
# create grid for different subplots
spec = gridspec.GridSpec(ncols=1, nrows=3,
                         # wspace=0.02,  width_ratios=[2, 2,2],
                         hspace=0.1, height_ratios=[4, 1, 1])
ax = fig.add_subplot(spec[0])
sm = ax.contourf(Z.T, cmap='Purples',extent=[xmin, xmax, ymin, ymax], 
                  levels = 10, vmin = 0, vmax = 80 )
sm1 = ax.contour(Z1.T, cmap='cmr.ember_r', extent=[xmin1, xmax1, ymin1, ymax1], 
                  levels = 10, vmin = 0, vmax = 80 )

# ax1.plot(X1, kurt1, 'k.', markersize=2)

ax.set(ylim = (-4.5,-1.5), xlim=(-0.015, 0.015), xticks = np.arange(-0.015, 0.0151, 0.005), 
        xticklabels = np.round(np.arange(-0.015, 0.0151, 0.005)*1000,2))# plt.colorbar(sm)
# plt.colorbar(sm)
plt.clabel(sm1, inline=True, fontsize=8, colors='k', fmt='%1.2f')
# ax.set_xlabel(r'$Wind\ Speed\ (m\ s^{-1})$', fontsize =14)
ax.set(xticklabels=[])
ax.set_ylabel(r'$log(\varepsilon)\ (m^2\ s^{-3})$', fontsize =14)
ax.tick_params(axis = 'both', labelsize = 14)
cbar_ax = fig.add_axes([0.9, 0.56, 0.02, 0.4])  # Adjust position and size of the colorbar
sm = plt.cm.ScalarMappable(cmap="Purples", norm=plt.Normalize(vmin=0, vmax=80))
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax, ticks=np.linspace(0, 80, 10), boundaries=np.linspace(0, 80, 10))


ax1 = fig.add_subplot(spec[1])

ax1.plot(X18, skew18, 'k', )
ax1.plot(X45, skew45, 'r')
ax1.tick_params(axis = 'y', labelsize = 14)

ax1.set(xlim = (-0.015, 0.015), xticklabels = [], ylim = (0,40))
ax1.set_ylabel('Skewness', fontsize=14)
 
ax2 = fig.add_subplot(spec[2])

ax2.plot(X45, kurt18, 'k')
ax2.plot(X45, kurt45, 'r')
ax2.set(xlim = (-0.015, 0.015), xticks = np.arange(-0.015, 0.0151, 0.005), 
        xticklabels = np.round(np.arange(-0.015, 0.0151, 0.005)*1000,2), ylim = (-50, 2200),
        yticks = np.arange(0,2200, 1000), yticklabels=np.arange(0,2200, 1000)/1000)
ax2.tick_params(axis = 'both', labelsize = 14)
ax2.set_xlabel(r'$Wind\ Shear\ (s^{-1})$', fontsize= 16)
ax2.set_ylabel('Kurtosis', fontsize=14)

c = ['k', 'r',]
l1 = [Line2D([0], [0], color=c, linewidth=2, linestyle='-') for c in c[:2]]
labels = [ 'Mid-Troposphere', 'Lower Troposphere',]
fig.legend(l1, labels, loc='center', fontsize =11, bbox_to_anchor=(0.63, 0.38), ncol=2)


fig.text(0.1, 0.225, r'$\times\ 10^3$')
fig.text(0.85, 0.025, r'$\times\ 10^{-3}$')
fig.subplots_adjust(top = 0.99, right = 0.89, bottom =0.08, left =0.11,)

fig.text(0.13, 0.95, '(a)', fontsize = 16)
fig.text(0.13, 0.35, '(b)', fontsize = 16)
fig.text(0.13, 0.19, '(c)', fontsize = 16)
fig.savefig('/home/ahana/Documents/pp4/fig/paper/tke_shear_gkde_kur_skew.jpg', dpi = 300)

#%%

x = a45.w.values[~np.isnan(a45.w.values)]
y = np.log10(a45.tke).values[~np.isnan(np.log10(a45.tke).values)]

xmin = -0.5
xmax = 0.5
ymin = -8.3
ymax = -0.5

X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])

values = np.vstack([x, y])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)

x1 = a18.w.values[~np.isnan(a18.w.values)]
y1 = np.log10(a18.tke).values[~np.isnan(np.log10(a18.tke).values)]

xmin1 = -0.5
xmax1 = 0.5

ymin1 =  -8.3
ymax1 = -0.5

X1,  Y1 = np.mgrid[xmin1 : xmax1 : 100j, ymin1 : ymax1 : 100j]
positions1 = np.vstack([X1.ravel(), Y1.ravel()])

values1 = np.vstack([x1, y1])
kernel1 = stats.gaussian_kde(values1)
kurt1 = stats.kurtosis(values1)

Z1 = np.reshape(kernel1(positions1).T, X1.shape)


x1 = a18.w.values[~np.isnan(a18.w.values)]
y1 = np.log10(a18.tke).values[~np.isnan(np.log10(a18.tke).values)]

xmin1 = -0.5
xmax1 = 0.5
X18 = np.linspace(xmin1, xmax1, 100)
kurt18 = np.zeros(100)
skew18 = np.zeros(100)
for i in range(len(X)-1):
    tk = a18.where((a18.w >= X18[i]) & (a18.w < X18[i+1])).tke.values
    tk = tk[~np.isnan(tk)]
    kurt18[i] = stats.kurtosis(tk, nan_policy='omit')
    skew18[i] = stats.skew(tk, nan_policy='omit')
    
x2 = a45.w.values[~np.isnan(a45.w.values)]
y2 = np.log10(a45.tke).values[~np.isnan(np.log10(a45.tke).values)]


xmin2 = -0.5
xmax2 = 0.5
X45 = np.linspace(xmin2, xmax2, 100)
kurt45 = np.zeros(100)
skew45 = np.zeros(100)
for i in range(len(X)-1):
    tk = a45.where((a45.w >= X45[i]) & (a45.w < X45[i+1])).tke.values
    tk = tk[~np.isnan(tk)]
    kurt45[i] = stats.kurtosis(tk, nan_policy='omit')
    skew45[i] = stats.skew(tk, nan_policy='omit')
    
#%%


fig = plt.figure(figsize = (8, 12))
 
# to change size of subplot's
# set height of each subplot as 8
fig.set_figheight(8)
 
# set width of each subplot as 8
fig.set_figwidth(8)
 
# create grid for different subplots
spec = gridspec.GridSpec(ncols=1, nrows=3,
                         # wspace=0.02,  width_ratios=[2, 2,2],
                         hspace=0.1, height_ratios=[4, 1, 1])
ax = fig.add_subplot(spec[0])
sm = ax.contourf(Z.T, cmap='Purples', extent=[xmin, xmax, ymin, ymax],
                  vmin = 0, vmax = 9, levels = 10,)
sm1 = ax.contour(Z1.T, cmap='cmr.ember_r',  extent=[xmin1, xmax1, ymin1, ymax1], 
                  vmin = 0, vmax = 9, levels = 10,)

# ax.plot(x, y, 'k.', markersize=2)

ax.set(ylim  =(-4,-2), xlim = (-0.1,0.3), xticks = np.arange(-0.1, 0.31, 0.05), 
        xticklabels = [])
# plt.colorbar(sm)
plt.clabel(sm1, inline=True, fontsize=8, colors='k', fmt='%1.2f')
ax.set_ylabel(r'$log(\varepsilon)\ (m^2\ s^{-3})$', fontsize =14)
ax.tick_params(axis = 'both', labelsize = 14)

cbar_ax = fig.add_axes([0.95, 0.45, 0.02, 0.5])  # Adjust position and size of the colorbar
sm = plt.cm.ScalarMappable(cmap="Purples", norm=plt.Normalize(vmin=0, vmax=9))
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax, ticks=np.linspace(0, 9, 10), boundaries=np.linspace(0, 9, 10),)


ax1 = fig.add_subplot(spec[1])

ax1.plot(X18, skew18, 'k', )
ax1.plot(X45, skew45, 'r')
ax1.tick_params(axis = 'y', labelsize = 14)

ax1.set(xlim = (-0.1, 0.3), xticks = np.arange(-0.1, 0.31, 0.05), xticklabels = [], yticks = np.arange(0,90,30))
ax1.set_ylabel('Skewness', fontsize=14)
 
ax2 = fig.add_subplot(spec[2])

ax2.plot(X45, kurt45, 'r')
ax2.plot(X18, kurt18, 'k')
ax2.set(xlim = (-0.1,0.3),  xticks = np.arange(-0.1, 0.31, 0.05), ylim=(-10,3300), 
        yticks = np.arange(0, 3100, 1500), yticklabels = (np.arange(0, 3100, 1500)/100).astype('int'))
ax2.tick_params(axis = 'both', labelsize = 14)
ax2.set_xlabel(r'$Vertical\ Wind\ Speed\ (m\ s^{-1})$', fontsize= 16)
ax2.set_ylabel('Kurtosis', fontsize=14)

c = ['k', 'r',]
l1 = [Line2D([0], [0], color=c, linewidth=2, linestyle='-') for c in c[:2]]
labels = [ 'Mid-Troposphere', 'Lower Troposphere',]
fig.legend(l1, labels, loc='center', fontsize =11, bbox_to_anchor=(0.63, 0.38), ncol=2)


fig.text(0.1, 0.225, r'$\times\ 10^2$')
fig.text(0.85, 0.025, r'$\times\ 10^{-2}$')

fig.text(0.14, 0.95, '(a)', fontsize = 16)
fig.text(0.14, 0.35, '(b)', fontsize = 16)
fig.text(0.14, 0.19, '(c)', fontsize = 16)
fig.subplots_adjust(top = 0.99, right = 0.94, bottom =0.08, left =0.13,)

fig.savefig('/home/ahana/Documents/pp4/fig/paper/tke_w_gkde_kur_skew.jpg', dpi = 300)
#%%
jjas45 = a45.sel(time=(a45["time.month"] >=6)&(a45["time.month"] <10))  #a45.sel(time = jjas(a45['time.month']))
djf45 = a45.sel(time = djf(a45['time.month']))          #a45.sel(time=(a45["time.month"] <3)&(a45["time.month"] ==12))
mam45 = a45.sel(time=(a45["time.month"] >=3)&(a45["time.month"] <6))   #a45.sel(time = mam(a45['time.month']))
ON45 = a45.sel(time=(a45["time.month"] >=11) )   #a45.sel(time = ON(a45['time.month']))

jjas18 = a18.sel(time=(a18["time.month"] >=6)&(a18["time.month"] <10))  #a18.sel(time = jjas(a18['time.month']))
djf18 = a18.sel(time = djf(a18['time.month']))
mam18 = a18.sel(time=(a18["time.month"] >=3)&(a18["time.month"] <6))   #a18.sel(time = mam(a18['time.month']))
ON18 = a18.sel(time=(a18["time.month"] >=11))    #a18.sel(time = ON(a18['time.month']))


#%%
s45 = [jjas45, djf45, mam45, ON45]
s18 = [jjas18, djf18, mam18, ON18]
lbl = ['JJAS', 'DJF', 'MAM', 'ON']
variables = ['wnd', 'duz', 'w']
#%%

fig, axes = plt.subplots(4, 3, figsize=(15, 16), sharey=True)  # 4 rows, 3 columns

# Iterate through rows (seasons) and columns (variables)
for i, season in enumerate(s45):
    for j, variable in enumerate(variables):
            
        if (j ==0):
            xj = s45[i][variables[j]].values[~np.isnan(s45[i][variables[j]].values)]
            yj = np.log10(s45[i].tke).values[~np.isnan(np.log10(s45[i].tke).values)]
            
            xminj = 0
            xmaxj = 30
            
            yminj = -8 #np.nanmin(yj)
            ymaxj = -0.5
            print (xminj, xmaxj, yminj, ymaxj)
            
            Xj, Yj = np.mgrid[xminj : xmaxj : 100j, yminj : ymaxj : 100j]
            positionsj = np.vstack([Xj.ravel(), Yj.ravel()])
            
            valuesj = np.vstack([xj, yj])
            kernelj = stats.gaussian_kde(valuesj)
            Zj = np.reshape(kernelj(positionsj).T, Xj.shape)
            
            x1j = s18[i].wnd.values[~np.isnan(s18[i].wnd.values)]
            y1j = np.log10(s18[i].tke).values[~np.isnan(np.log10(s18[i].tke).values)]
            
            xmin1j = 0
            xmax1j = 30
            
            ymin1j = -8
            ymax1j = -0.5
            
            X1j,  Y1j = np.mgrid[xmin1j : xmax1j : 100j, ymin1j : ymax1j : 100j]
            positions1j = np.vstack([X1j.ravel(), Y1j.ravel()])
            
            values1j = np.vstack([x1j, y1j])
            kernel1j = stats.gaussian_kde(values1j)
            Z1j = np.reshape(kernel1j(positions1j).T, X1j.shape)
        
            sm = axes[i, j].contourf(Zj.T, cmap='Purples', extent=[xminj, xmaxj, yminj, ymaxj], 
                           vmin = 0, vmax = 0.114, levels = 10, )
            sm1 = axes[i, j].contour(Z1j.T, cmap='cmr.ember_r', extent=[xmin1j, xmax1j, ymin1j, ymax1j], 
                                vmin = 0, vmax = 0.114, levels = 10, )
            axes[i, j].set( ylim = (-5,-1), xlim=(0, 20), xticks = np.arange(0, 21, 4))
            plt.clabel(sm1, inline=True, fontsize=8, colors='k', fmt='%1.2f')
            axes[i, j].set_ylabel(r'$log(\varepsilon)\ (m^2\ s^{-3})$', fontsize = 12)
            if i ==3:
                axes[i, j].set_xlabel(r'$Wind\ Speed\ (m\ s^{-1})$', fontsize = 12)
            else:
                axes[i,j].set(xticklabels = [])
            cbar_ax = fig.add_axes([0.08, 0.035, 0.25, 0.015])  # Adjust position and size of the colorbar
            sm = plt.cm.ScalarMappable(cmap="Purples", norm=plt.Normalize(vmin=0, vmax=0.114))
            sm.set_array([])
            fig.colorbar(sm, cax=cbar_ax, ticks= np.round(np.linspace(0, 0.114, 10),2), orientation = 'horizontal',
                         boundaries= np.linspace(0, 0.114, 10))
            
        if (j ==1):
            xj = s45[i][variables[j]].values[~np.isnan(s45[i][variables[j]].values)]
            yj = np.log10(s45[i].tke).values[~np.isnan(np.log10(s45[i].tke).values)]
            
            xminj = -0.18
            xmaxj = 0.18
            
            yminj = -8.3
            ymaxj = -0.5
            # print (xmaxj, yminj, ymaxj)
            Xj, Yj = np.mgrid[xminj : xmaxj : 100j, yminj : ymaxj : 100j]
            positionsj = np.vstack([Xj.ravel(), Yj.ravel()])
            
            valuesj = np.vstack([xj, yj])
            kernelj = stats.gaussian_kde(valuesj)
            Zj = np.reshape(kernelj(positionsj).T, Xj.shape)
            
            x1j = s18[i][variables[j]].values[~np.isnan(s18[i][variables[j]].values)]
            y1j = np.log10(s18[i].tke).values[~np.isnan(np.log10(s18[i].tke).values)]
            
            xmin1j = -0.18
            xmax1j = 0.18
            
            ymin1j = -8.3
            ymax1j = -0.5
            print (xmax1j, ymin1j, ymax1j)
            
            X1j,  Y1j = np.mgrid[xmin1j : xmax1j : 100j, ymin1j : ymax1j : 100j]
            positions1j = np.vstack([X1j.ravel(), Y1j.ravel()])
            
            values1j = np.vstack([x1j, y1j])
            kernel1j = stats.gaussian_kde(values1j)
            Z1j = np.reshape(kernel1j(positions1j).T, X1j.shape)
            
            sm = axes[i, j].contourf(Zj.T, cmap='Purples', extent=[xminj, xmaxj, yminj, ymaxj], 
                          vmin = 0, vmax = 100, levels = 10, )
            sm1 = axes[i, j].contour(Z1j.T, cmap='cmr.ember_r', extent=[xmin1j, xmax1j, ymin1j, ymax1j], 
                              vmin = 0, vmax = 100, levels = 10, )
            plt.clabel(sm1, inline=True, fontsize=8, colors='k', fmt='%1.2f')
            axes[i, j].set( ylim = (-4.5,-1), xlim=(-0.015,0.016), xticks = np.arange(-0.015, 0.016, 0.005),)
            if i ==3:
                axes[i, j].set_xlabel(r'$Wind\ Shear\ (s^{-1})$', fontsize = 12)
                axes[i, j].set(xticks = np.arange(-0.015, 0.016, 0.005), 
                       xticklabels = np.round(np.arange(-0.015, 0.016, 0.005)*1000,2))
            else:
                axes[i,j].set(xticks = np.arange(-0.015, 0.016, 0.005), 
                       xticklabels = [])
            cbar_ax = fig.add_axes([0.39, 0.035, 0.25, 0.015])  # Adjust position and size of the colorbar
            sm = plt.cm.ScalarMappable(cmap="Purples", norm=plt.Normalize(vmin=0, vmax=100))
            sm.set_array([])
            fig.colorbar(sm, cax=cbar_ax, ticks= np.round(np.linspace(0, 100, 10),2), orientation = 'horizontal',
                         boundaries=np.linspace(0, 100, 10))
            
            
        if (j == 2):
            xj = s45[i][variables[j]].values[~np.isnan(s45[i][variables[j]].values)]
            yj = np.log10(s45[i].tke).values[~np.isnan(np.log10(s45[i].tke).values)]
            # q = stats.linregress(xj, yj)
            # print (str(lbl[i])+'low',q[0])
            
            xminj = -0.5
            xmaxj = 0.5

            yminj = -8.3
            ymaxj = -0.5
            print (xmaxj, yminj, ymaxj)
            Xj, Yj = np.mgrid[xminj : xmaxj : 100j, yminj : ymaxj : 100j]
            positionsj = np.vstack([Xj.ravel(), Yj.ravel()])
            
            valuesj = np.vstack([xj, yj])
            kernelj = stats.gaussian_kde(valuesj)
            Zj = np.reshape(kernelj(positionsj).T, Xj.shape)
            
            x1j = s18[i][variables[j]].values[~np.isnan(s18[i][variables[j]].values)]
            y1j = np.log10(s18[i].tke).values[~np.isnan(np.log10(s18[i].tke).values)]
            # q1 = stats.linregress(x1j, y1j)
            # print (str(lbl[i])+'mid', q1[0])
            
            xmin1j = -0.5
            xmax1j = 0.5
            
            ymin1j = -8.3
            ymax1j = -0.5
            
            X1j,  Y1j = np.mgrid[xmin1j : xmax1j : 100j, ymin1j : ymax1j : 100j]
            positions1j = np.vstack([X1j.ravel(), Y1j.ravel()])
            
            values1j = np.vstack([x1j, y1j])
            kernel1j = stats.gaussian_kde(values1j)
            Z1j = np.reshape(kernel1j(positions1j).T, X1j.shape)
            
            sm = axes[i,j].contourf(Zj.T, cmap='Purples',extent=[xminj, xmaxj, yminj, ymaxj], 
                            vmin = 0, vmax = 8, levels = 10, )
            sm1 = axes[i,j].contour(Z1j.T, cmap='cmr.ember_r', extent=[xmin1j, xmax1j, ymin1j, ymax1j], 
                                vmin = 0, vmax = 8, levels = 10, )
            plt.clabel(sm1, inline=True, fontsize=8, colors='k', fmt='%1.2f')
            axes[i, j].set(ylim = (-4.5, -1), xlim=(-0.005, 0.25))
            if i ==3:
                axes[i, j].set_xlabel(r'$Vertical\ Wind\ Speed\ (m\ s^{-1})$', fontsize = 12)
            else:
                axes[i,j].set(xticklabels = [])
            cbar_ax = fig.add_axes([0.69, 0.035, 0.25, 0.015])  # Adjust position and size of the colorbar
            sm = plt.cm.ScalarMappable(cmap="Purples", norm=plt.Normalize(vmin=0, vmax=8))
            sm.set_array([])
            cl = fig.colorbar(sm, cax=cbar_ax, ticks= np.round(np.linspace(0, 8, 10),2), orientation = 'horizontal',
                              boundaries=np.linspace(0, 8, 10))
            
# fig.tight_layout()

fig.text(0.97, 0.85, 'JJAS', rotation = 90, fontsize =16)
fig.text(0.97, 0.63, 'DJF', rotation = 90, fontsize =16)
fig.text(0.97, 0.4, 'MAM', rotation = 90, fontsize =16)
fig.text(0.97, 0.19, 'ON', rotation = 90, fontsize =16)
fig.subplots_adjust(top = 0.99, right = 0.95, bottom =0.085, left =0.07, hspace = 0.02, wspace = 0.02)

fig.savefig('/home/ahana/Documents/pp4/fig/paper/tke_seas', dpi = 300)
#%%


da3 = np.log10(a45.tke.sel(level = slice(0, 3)).resample(time='1M').mean('time')).mean('level')
da6 = np.log10(a18.tke.sel(level = slice(3, 6)).resample(time='1M').mean('time')).mean('level')
da9 = np.log10(a18.tke.sel(level = slice(6, 9)).resample(time='1M').mean('time')).mean('level')
da12 = np.log10(a18.tke.sel(level = slice(9, 12)).resample(time='1M').mean('time')).mean('level')

f = [da3, da6, da9, da12]
lb = ['0-3', '3-6', '6-9','9-12']
x=np.arange(len(a45.time))
x1=np.arange(len(a18.time))


fig = plt.figure(figsize=(15,10))
for k in range(len(f)):
    
    tk = f[k].values
    ax = fig.add_subplot(2,2,k+1)
    # if k < 5:
    x=np.arange(len(f[k].time)) 
    p=np.polyfit(x, tk, 1)
    q = linregress(x, tk)
    confidence_level = 0.9
    #Degrees of freedom (for simple linear regression, it's n - 2)
    degrees_of_freedom = len(x) - 2
    critical_t_value = t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
    print (critical_t_value)
    tv = q.slope/q.stderr
    scientific_notation = "{:e}".format(q[0])
    parts = scientific_notation.split('e')
    fr = "{:.1f} * 10^{}".format(float(parts[0]), int(parts[1]))
    
    if abs(tv)>critical_t_value:
        print(lb[k], q.slope, q.pvalue, tv, "signifiant")
        ax.plot(x ,tk, 'k', label = lb[k]+' km')
        ax.plot(x, q.intercept + q.slope*x, 'r', label = r'$slope\ =\ $' + fr)
    else:
        print (lb[k], q.slope,  q.pvalue, tv, "not significant")
        ax.plot(x ,tk, 'k', label = lb[k]+' km',)
        ax.plot(x, q.intercept + q.slope*x, 'r', label = r'$slope\ =\ $' + fr, linestyle = '--')
    # if q.pvalue< 0.1:
    #     print (lb[k], "significant")
    # else:
    #     print (lb[k], "not significant")
    # print ("pvalue = "+ str(q[3]))
    
    mm = ax.plot([], []  , c= 'white', label = "Confidence level = "+ str(np.round((1-q[3])*100,1)),)
    tim = [str(k)[:7] for k in f[k].time.values]
    if k%2 ==0:
        ax.set(ylim=(-4,-2), yticks = np.arange(-4,-1.9,0.5))
        ax.set_ylabel(r'$log(\varepsilon)\ (m^2\ s^{-3})$', fontsize= 15)
    else:
        ax.set(ylim=(-4,-2), yticklabels=[])#
    if k>1:
        ax.set(xticks = x[::12], xticklabels= tim[::12])
    else:
        ax.set(xticklabels= [])
    # ax.legend(mm, loc = 'lower right', fontsize = 14, handlelength = 0)
    ax.legend(loc = 'lower right', fontsize = 14,)
    ax.tick_params(axis = 'both', labelsize = 14)
fig.tight_layout()
fig.subplots_adjust(top = 0.99, right = 0.99, bottom =0.08, left =0.06, wspace = 0.06)
fig.savefig('/home/ahana/Documents/pp4/fig/tke_mean_layer3km_lr(0-3)_trend.jpg', dpi = 300)
