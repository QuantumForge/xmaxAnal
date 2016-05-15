# xmaxAnal
Tests of statistical compatibility between competing analysis efforts of UHECR Xmax distributions

cvm_2samp.py: class to perform a CvM 2 sample test. Computes p-values based on limiting statistics as sample sizes approach infinity. ECDFs can be grabbed from the class as plotted as well.

xmaxCompat.py: class to read in 2 sets of Xmax data and plot distributions, sample those distributions in case they are weighted, and perform a CvM 2 sample test on them. Data shifting is there as well. We have to sample the Xmax distributions if one person has a weighted sample. That weighted sample would distort the ECDF. So fit the weighted sample, draw a sample of Xmax from the fits, and perform statistical compatibility test on those draws.

xmaxFit.py: fits a distribution of xmax data by histogramming into user defined bins, applying weights if provided, then fitting to an analytical function which is a convolution of a normal and exponential to fit the tails.


