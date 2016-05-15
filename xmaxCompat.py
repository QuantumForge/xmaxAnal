#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.integrate as integrate

import xmaxFit
import cvm_2samp

class xmaxCompat:
    def __init__(self,
            fnameHanlon = 'dump_hanlon_xmaxAcceptance_reconOnly.txt.bz2',
            fnameIkeda  = 'dump_ikeda_xmaxAcceptance.txt.bz2'):
        self.dfHanlon = pd.read_csv(fnameHanlon,
                  delim_whitespace=True, header=None, names=['date', 'time',
                      'weight', 'eThrown', 'xmaxThrown', 'eRecon', 'xmaxRecon'],
                  na_values=['-1'], parse_dates=[[0, 1]])
        self.dfIkeda = pd.read_csv(fnameIkeda,
                delim_whitespace=True, header=None, names=['date', 'time',
                    'weight', 'eThrown', 'xmaxThrown', 'eRecon', 'xmaxRecon'],
                na_values=['-1'], parse_dates=[[0, 1]])

    def sample(self, func, funcArgs, nsamp=1000, dataRange=[500., 1300.]):
        """Sample an xmax function by acceptance-rejection method and return
        nsamp xmax values randomly distributed by func."""

        # scan over range for the function maximum
        x = np.linspace(dataRange[0], dataRange[1], 1000)
        fmax = np.max(func(x, *funcArgs))
        # list of sampled xmax
        val = []
        # acceptance-rejection method sampling. I need to learn proper MCMC
        # techniques...
        n = 0
        while (True):
            r = np.random.rand(2)
            rx = dataRange[0] + (dataRange[1] - dataRange[0])*r[0]
            ry = fmax*r[1]    # lower bound is 0
            if (ry < func(rx, *funcArgs)):
                val.append(rx)
                n += 1
            if (n >= nsamp):
                return np.array(val)

    def main(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True)

        hdata = self.dfHanlon.xmaxThrown[(self.dfHanlon['eThrown']>=18.2) & \
                (self.dfHanlon['eThrown']<18.4)]
        hxmf = xmaxFit.xmaxFit(hdata, 18.3)
        #print hxmf.pname
        #print hxmf.pfit
        #print hxmf.perr

        ax1.hist(hdata, bins=80, range=[500., 1300.])
        funcX = np.linspace(500., 1300., 100)
        funcY = hxmf.func(funcX, *hxmf.pfit)

        ax1.plot(funcX, funcY, linewidth=2., linestyle='dashed',
                color='black')
        ax1.set_xlabel('$X_{\mathrm{max}}$ (g/cm$^{2}$)')
        ax1.set_ylabel('$N$')
        ax1.grid()


        idata = self.dfIkeda.xmaxThrown[(self.dfIkeda['eThrown']>=18.2) & \
                (self.dfIkeda['eThrown']<18.4) & \
                (self.dfIkeda['eRecon'] > 0)]
        iweight = self.dfIkeda.weight[(self.dfIkeda['eThrown']>=18.2) & \
                (self.dfIkeda['eThrown']<18.4) & \
                (self.dfIkeda['eRecon'] > 0)]

        ixmf = xmaxFit.xmaxFit(idata, 18.3, weights=iweight)
        #print ixmf.pname
        #print ixmf.pfit
        #print ixmf.perr

        ax2.hist(idata, bins=80, range=[500., 1300.])
        funcX = np.linspace(500., 1300., 100)
        funcY = ixmf.func(funcX, *ixmf.pfit)

        ax2.plot(funcX, funcY, linewidth=2., linestyle='dashed',
                color='black')
        ax2.set_xlabel('$X_{\mathrm{max}}$ (g/cm$^{2}$)')
        ax2.set_ylabel('$N$')
        ax2.grid()

        hxmaxpdf = self.sample(hxmf.func, hxmf.pfit, nsamp=100000)
        ax3.hist(hxmaxpdf, bins=80, range=[500., 1300.])
        ax3.set_xlabel('$X_{\mathrm{max}}$ (g/cm$^{2}$)')
        ax3.set_ylabel('$N$')
        ax3.grid()
        
        ixmaxpdf = self.sample(ixmf.func, ixmf.pfit, nsamp=100000)
        ax4.hist(ixmaxpdf, bins=80, range=[500., 1300.])
        ax4.set_xlabel('$X_{\mathrm{max}}$ (g/cm$^{2}$)')
        ax4.set_ylabel('$N$')
        ax4.grid()
        
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    xmaxCompat().main()
