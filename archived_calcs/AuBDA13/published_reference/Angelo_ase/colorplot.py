import numpy as np
import pickle

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter

# inputs and bfs order changes depending on script used: subsort.py OR subdiag.py 
sorting = False

h_scatt, s_scatt = pickle.load(open('scatt_hs_lcao.pckl', 'rb'))
h_sub,   s_sub   = pickle.load(open('scatt_hs_sub.pckl',  'rb'))
h_pz,    s_pz    = pickle.load(open('scatt_hs_pz.pckl',   'rb'))
h_pzd,   s_pzd   = pickle.load(open('scatt_hs_pzd.pckl',  'rb'))

if sorting:
    h_sort,  s_sort  = pickle.load(open('scatt_hs_sort.pckl', 'rb'))

nbfs = h_scatt.shape[0]
nm   = 6*13+2*13+8*5
nc   = 6*13 
ni   = 612
nf   = ni + nm

# custom colorbar
clist = [(1,1,1), (1,0,0)]
nbins = 100
cmap = LinearSegmentedColormap.from_list('custom', clist, N=nbins)

# plot
fig, axes = plt.subplots(nrows=2, ncols=2)

axes[0,0].title.set_text('molecule LCAO')
axes[0,1].title.set_text('molecule subdiagonalized')
axes[1,0].title.set_text('molecule C-pz')
axes[1,1].title.set_text('molecule C-pz+d')

cbmin=0.01
cbmax=10

axes[0,0].imshow(abs(h_scatt[ni:nf,ni:nf].real), cmap=cmap, norm=LogNorm(vmin=cbmin, vmax=cbmax))
axes[0,1].imshow(abs(h_sub[ni:nf,ni:nf].real),   cmap=cmap, norm=LogNorm(vmin=cbmin, vmax=cbmax))

if sorting:
    axes[1,0].imshow(abs(h_pz[-nm:,-nm:].real),      cmap=cmap, norm=LogNorm(vmin=cbmin, vmax=cbmax))
    axes[1,1].imshow(abs(h_pzd[-nm:,-nm:].real),     cmap=cmap, norm=LogNorm(vmin=cbmin, vmax=cbmax))
else:
    axes[1,0].imshow(abs(h_pz[ni:nf,ni:nf].real),    cmap=cmap, norm=LogNorm(vmin=cbmin, vmax=cbmax))
    axes[1,1].imshow(abs(h_pzd[ni:nf,ni:nf].real),   cmap=cmap, norm=LogNorm(vmin=cbmin, vmax=cbmax))


im = cm.ScalarMappable(cmap=cmap)
im.set_clim(vmin=cbmin, vmax=cbmax)
formatter = LogFormatter(10, labelOnlyBase=False)
plt.colorbar(im, ax=axes.ravel().tolist(), ticks=[0.01,10], format=formatter)


plt.show()


