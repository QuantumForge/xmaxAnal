#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import xmaxFit

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

    def main(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)

        hdata = self.dfHanlon.xmaxThrown[(self.dfHanlon['eThrown']>=18.2) & \
                (self.dfHanlon['eThrown']<18.4)]
        hxmf = xmaxFit.xmaxFit(hdata, 18.3)
        print hxmf.pname
        print hxmf.pfit
        print hxmf.perr

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
        print ixmf.pname
        print ixmf.pfit
        print ixmf.perr

        ax2.hist(idata, bins=80, range=[500., 1300.])
        funcX = np.linspace(500., 1300., 100)
        funcY = ixmf.func(funcX, *ixmf.pfit)

        ax2.plot(funcX, funcY, linewidth=2., linestyle='dashed',
                color='black')
        ax2.set_xlabel('$X_{\mathrm{max}}$ (g/cm$^{2}$)')
        ax2.set_ylabel('$N$')
        ax2.grid()

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    xmaxCompat().main()
