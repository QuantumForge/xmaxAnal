# xmaxAnal
Tests of statistical compatibility between competing analysis efforts of UHECR Xmax distributions

cvm_2samp.py: class to perform a CvM 2 sample test. Computes p-values based on limiting statistics as sample sizes approach infinity. ECDFs can be grabbed from the class as plotted as well.

xmaxCompat.py: class to read in 2 sets of Xmax data and plot distributions, sample those distributions in case they are weighted, and perform a CvM 2 sample test on them. Data shifting is there as well. We have to sample the Xmax distributions if one person has a weighted sample. That weighted sample would distort the ECDF. So fit the weighted sample, draw a sample of Xmax from the fits, and perform statistical compatibility test on those draws.

xmaxFit.py: fits a distribution of xmax data by histogramming into user defined bins, applying weights if provided, then fitting to an analytical function which is a convolution of a normal and exponential to fit the tails.

xmaxCompatSession.ipynb: jupyter notebook session of one run through the data


## How to run:
The most useful way is to start ipython
before starting be sure to have the two data files required: *dump_hanlon_xmaxAcceptance_reconOnly.txt.bz2* and *dump_ikeda_xmaxAcceptance.txt.bz2*.

import the main class
> import xmaxCompat

instantiate
> x = xmaxCompat.xmaxCompat()

now using the main method, different energy bins can be examined
> x.main(18.2, 18.4)

> x.main(18.4, 18.6)

> x.main(18.6, 18.8)

etc.
the purpose of the program is to compare two independently generated Monte Carlo data sets which are supposed to be draw from the same input distribution, then after detector effects and event reconstruction, examine if the data appear to be compatible using a Cramer-von Mises test I can measure the compatibility of the distributions.
