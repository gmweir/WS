# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 11:26:46 2017

@author: gawe
"""

import numpy as _np
import query_programs as _qp
import matplotlib.pyplot as _plt
import pybaseutils as _pyb

shot = 171123035

# ==== #

print('getting ECRH')
ECRH = _qp.get_archive_byname('ECRH_tot', shot)

t_p = 1e-9*_np.asarray(ECRH[1], dtype=_np.float64)
ECRH = _np.asarray(ECRH[0], dtype=_np.float64)

ECRH *= 1e-3 # MW

# ==== #

print('getting line-integrated (fast) density')
nedl = _qp.get_archive_byname('density', shot)

t_n = 1e-9*_np.asarray(nedl[1], dtype=_np.float64)
nedl = _np.asarray(nedl[0], dtype=_np.float64)

inds = _np.where(t_n>-0.01)[0]

nedl = _qp.densityscaler(nedl, shot)
nedl *= 1e-19
Fs = 1.0/(t_n[1]-t_n[0])

def unwrap_phase(sigin, Fs, mult=10.0, nmonti=1, nlevel=1e-3):
    sig = sigin.copy()
    rtol = 0.25
    for ii in range(nmonti):
        print(ii)
        for scal in [1.0, 0.5, 0.25]:
            for itol in [ii+1 for ii in range(mult, mult*5, mult)]:
                print('x')
                if mult>1:    sig = _pyb.upsample(sig, Fs, mult*Fs)
                print('y')
                sig = sig.flatten()
                if nlevel>0:
                    sig += _np.random.normal(0.0, nlevel, sig.shape)
                print('z')
                sig = _pyb.fft_analysis.unwrap_tol(sig.copy(), scal, rtol=rtol, itol=itol)
                print('a')
                if mult>1:    sig = _pyb.downsample(sig, mult*Fs, Fs)
                print('b')
                sig = sig.flatten()
                sig -= sig[inds[0]]
                print('c')
            # end for
        # end for
    # end for

#    sig = _pyb.upsample(sig, Fs, 10.0*Fs)
#    sig = sig.flatten()
#    sig += _np.random.normal(0.0, 0.03, sig.shape)
#
#    rtol = 0.25
#    for direction in [1, -1]:
#        if direction<0:        sig = _np.flipud(sig)
#        for scal in [1.0, 0.5, 0.25]:
#            for itol in [ii+1 for ii in range(5)]:
#                sig = _pyb.fft_analysis.unwrap_tol(sig.copy(), scal, rtol=rtol, itol=itol)
#                sig -= sig[inds[0]]
#            # end for
#        # end for
#        if direction<0:        sig = _np.flipud(sig)
#    # end for
#
#    sig = _pyb.downsample(sig, 10.0*Fs, Fs)
#    sig = sig.flatten()
    #sig -= sig[-1]

    _plt.figure(), _plt.plot(t_n, sigin, 'b-', t_n, sig, 'r-')
    return sig
nedl = unwrap_phase(nedl, Fs, mult=int(1), nmonti=1, nlevel=0)

# ==== #

print('getting line-integrated (slow) density')
nedl_slow = _qp.get_archive_byname('density_slow', shot)

t_ns = 1e-9*_np.asarray(nedl_slow[1], dtype=_np.float64)
nedl_slow = _np.asarray(nedl_slow[0], dtype=_np.float64)
#nedl_slow = _pyb.fft_analysis.unwrap_tol(nedl_slow, scal=1.5, atol=0.005)
nedl_slow *= 1e-19

# ==== #

print('getting stored energy')
Wdia = _qp.get_archive_byname('Wdia', shot)

t_e = 1e-9*_np.asarray(Wdia[1], dtype=_np.float64)
Wdia = _np.asarray(Wdia[0], dtype=_np.float64)
Wdia *= 1e-3   #  MJ


# ==== #

print('getting Thomson density')
ne, tsch, tt = _qp.get_thomson_ne(shot)
print('getting Thomson temperature')
Te, tsch, tt = _qp.get_thomson_Te(shot)
tt *= 1e-9 # s
#ne *= 1e-19
#ne = ne.transpose()
#Te = Te.transpose()

# ==== #

_plt.figure()
_ax1 = _plt.subplot(4,1,1)
_plt.plot(t_p, ECRH, 'r-')
_ax2 = _plt.subplot(4,1,2, sharex=_ax1)
_plt.plot(t_e, Wdia, 'r-')
_ax3 = _plt.subplot(4,1,3, sharex=_ax1)
_plt.plot(t_n, nedl, 'b-')
_plt.plot(t_ns, nedl_slow, 'm-')
_plt.plot(tt, ne, 'o')
_ax4 = _plt.subplot(4,1,4, sharex=_ax1)
_plt.plot(tt, Te, 'o')
_plt.ylim((0,12))
