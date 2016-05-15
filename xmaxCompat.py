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
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = \
                plt.subplots(3, 2, sharex=True)

        # read in hanlon xmax data. my data is unweighted
        hdata = self.dfHanlon.xmaxThrown[(self.dfHanlon['eThrown']>=18.2) & \
                (self.dfHanlon['eThrown']<18.4)]
        print 'Hanlon Xmax'
        hxmf = xmaxFit.xmaxFit(hdata, 18.3)
        #print hxmf.pname
        #print hxmf.pfit
        #print hxmf.perr

        # plot weighed distribution and fit
        ax1.hist(hdata, bins=80, range=[500., 1300.])
        funcX = np.linspace(500., 1300., 100)
        funcY = hxmf.func(funcX, *hxmf.pfit)

        ax1.plot(funcX, funcY, linewidth=2., linestyle='dashed',
                color='black')
        ax1.set_xlabel('$X_{\mathrm{max}}$ (g/cm$^{2}$)')
        ax1.set_ylabel('$N$')
        ax1.grid()

        # read in ikeda xmax data. his data is weighted, so we must weight
        # the distribution, fit it, then sample it so we can generate an
        # undistorted ECDF.
        idata = self.dfIkeda.xmaxThrown[(self.dfIkeda['eThrown']>=18.2) & \
                (self.dfIkeda['eThrown']<18.4) & \
                (self.dfIkeda['eRecon'] > 0)]
        iweight = self.dfIkeda.weight[(self.dfIkeda['eThrown']>=18.2) & \
                (self.dfIkeda['eThrown']<18.4) & \
                (self.dfIkeda['eRecon'] > 0)]

        print '\n'
        print 'Ikeda Xmax'
        ixmf = xmaxFit.xmaxFit(idata, 18.3, weights=iweight)
        #print ixmf.pname
        #print ixmf.pfit
        #print ixmf.perr

        # plot the weighted distribution and fit
        ax2.hist(idata, bins=80, range=[500., 1300.])
        funcX = np.linspace(500., 1300., 100)
        funcY = ixmf.func(funcX, *ixmf.pfit)

        ax2.plot(funcX, funcY, linewidth=2., linestyle='dashed',
                color='black')
        ax2.set_xlabel('$X_{\mathrm{max}}$ (g/cm$^{2}$)')
        ax2.set_ylabel('$N$')
        ax2.grid()

        # now generate a sample of randomly drawn xmax from the fitted
        # distributions. this removes weighting bias
        hxmaxpdf = self.sample(hxmf.func, hxmf.pfit, nsamp=10000,
                dataRange=[500., 1300.])
        ax3.hist(hxmaxpdf, bins=80, range=[500., 1300.])
        ax3.set_xlabel('$X_{\mathrm{max}}$ (g/cm$^{2}$)')
        ax3.set_ylabel('$N$')
        ax3.grid()
        
        # now generate a sample of randomly drawn xmax from the fitted
        # distributions. this removes weighting bias
        ixmaxpdf = self.sample(ixmf.func, ixmf.pfit, nsamp=10000,
                dataRange=[500., 1300.])
        ax4.hist(ixmaxpdf, bins=80, range=[500., 1300.])
        ax4.set_xlabel('$X_{\mathrm{max}}$ (g/cm$^{2}$)')
        ax4.set_ylabel('$N$')
        ax4.grid()


        # means of the two sampled distributions
        mean_hxmaxpdf = np.mean(hxmaxpdf)
        mean_ixmaxpdf = np.mean(ixmaxpdf)
        shift = mean_hxmaxpdf - mean_ixmaxpdf
        print '\n'
        print ' <Hanlon Xmax> =', mean_hxmaxpdf
        print ' <Ikeda Xmax> =', mean_ixmaxpdf
        print 'shift =', shift

        # use a Cramer-von Mises 2 sample test to test the compatibility of the
        # data. there is most likely a systematic bias bewteen the distributions
        # so we may have to look at shifting them as well...
        cvm = cvm_2samp.cvm_2samp(hxmaxpdf, ixmaxpdf - shift)

        # evaluate the p-value.
        # H0: hxmaxpdf and ixmaxpdf are both drawn from the same parent
        # population
        # Ha: hxmaxpdf and ixmaxpdf are not both drawn from the same parent
        # population
        # cvm_2samp evaluates the test statistic for the limiting distribution
        # as the number of samples approaches infinity. report the true test
        # statistic measured and the p-value of the limiting distribution.
        # (we have many. many samples here)
        (T, _, p) = cvm.eval()
        print '\n'
        print 'CvM test statistic:', T
        print 'p-value:', p

        # this test is much more sensitive in the tails than I expected...

        # plot ECDFs of the sampled distributions. input xmax distributions
        # aren't sorted, so provide sorted lists for plotting.
        ax5.plot(np.sort(hxmaxpdf), cvm.ecdf_x)
        ax5.set_xlabel('$X_{\mathrm{max}}$ (g/cm$^{2}$)')
        ax5.set_ylabel('Cumulative probability')
        ax5.grid()
        ax6.plot(np.sort(ixmaxpdf), cvm.ecdf_y)
        ax6.set_xlabel('$X_{\mathrm{max}}$ (g/cm$^{2}$)')
        ax6.set_ylabel('Cumulative probability')
        ax6.grid()
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    xmaxCompat().main()
