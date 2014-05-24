# May 8, 2010

import numpy
import scipy.special
from matplotlib import use
use("Agg")
import matplotlib
import matplotlib.pyplot

def plplot(x, xmin, alpha, variable='x', p_value=None, png='plplot'):
    """
% PLPLOT visualizes a power-law distributional model with empirical data.
% 
%    PLPLOT(x, xmin, alpha) plots (on log axes) the data contained in x 
%    and a power-law distribution of the form p(x) ~ x**-alpha for 
%    x >= xmin. For additional customization, PLPLOT returns a pair of 
%    handles, one to the empirical and one to the fitted data series. By 
%    default, the empirical data is plotted as 'bo' and the fitted form is
%    plotted as 'k--'. PLPLOT automatically detects whether x is composed 
%    of real or integer values, and applies the appropriate plotting 
%    method. For discrete data, if min(x) > 50, PLFIT uses the continuous 
%    approximation, which is reliable in this regime.
%
%    Example:
%       xmin  = 5
%       alpha = 2.5
%       xc = xmin*(1-numpy.random.random(10000))**(-1/(alpha-1))
%       hc = plplot.plplot(xc, xmin, alpha)
%       xd = numpy.floor((xmin-0.5)*(1-numpy.random.random(10000))**(-1/(alpha-1))+0.5)
%       hd = plplot.plplot(xd, xmin, alpha)
%
%    See also PLFIT, PLVAR, PLPVA
%
    """

    x = numpy.array(x)
    x = x[x>0]
    
    # initialize storage for output handles
    h = []

    # estimate xmin and alpha, accordingly

    n = float(len(x))

    # continuous method
#    if min(x) > 50 or sum(x-numpy.floor(x)):
    if sum(x-numpy.floor(x)):
        c = numpy.r_[numpy.sort(x), numpy.arange(n, 0, -1)/n].reshape(2, n)
        q = numpy.sort(x[x >= xmin])
        cf = numpy.r_[q, (q/xmin)**(1-alpha)].reshape(2, len(q))
        cf[1, :] = cf[1, :] * max(c[1, c[0, :] >= xmin])

    # discrete method
    else:
        q = numpy.unique(x)
        c = numpy.histogram(x, numpy.r_[q, q[-1]+1])[0] / n
        c = numpy.r_[numpy.r_[q, q[-1]+1], 1 - numpy.r_[0, numpy.cumsum(c)]].reshape(2, len(q)+1)
        c = c[:, c[1, :] >= 1e-10]
        cf = (numpy.r_[xmin:q[-1]+1]**-alpha)/scipy.special.zeta(alpha, xmin)
        cf = numpy.r_[numpy.r_[xmin:q[-1]+2], 1 - numpy.r_[0, numpy.cumsum(cf)]].reshape(2, q[-1]+2-xmin)
        cf[1, :] = cf[1, :] * c[1, c[0, :] == xmin]

    #matplotlib.rc('font',**{'family':'serif','serif':['Palatino']))
    #matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    matplotlib.rc('font', family='serif')
    matplotlib.rc('text', usetex=True)

    matplotlib.pyplot.figure()
    h.append(matplotlib.pyplot.loglog(c[0, :], c[1, :], 'bo', markersize=8, markerfacecolor='w'))
    h.append(matplotlib.pyplot.loglog(cf[0, :], cf[1, :], 'k--', linewidth=2))
    xr  = (10**numpy.floor(numpy.log10(min(x))), 10**numpy.ceil(numpy.log10(max(x)))) # x limits
    xrt = numpy.arange(round(numpy.log10(xr[0])), round(numpy.log10(xr[1])) + 1, 2)
    if len(xrt) < 4:
        xrt = numpy.arange(round(numpy.log10(xr[0])), round(numpy.log10(xr[1])) + 1, 1)
    yr  = (10**numpy.floor(numpy.log10(min(c[1, :]))), 1) # y limits
    yrt = numpy.arange(round(numpy.log10(yr[0])), round(numpy.log10(yr[1])) + 1, 2)
    if len(yrt) < 4:
        yrt = numpy.arange(round(numpy.log10(yr[0])), round(numpy.log10(yr[1])) + 1, 1)
    matplotlib.pyplot.xlim(xr)
    matplotlib.pyplot.xticks(10**xrt, fontsize=16)
    matplotlib.pyplot.ylim(yr)
    matplotlib.pyplot.yticks(10**yrt, fontsize=16)
    matplotlib.pyplot.ylabel(r'$P(%s)$' % variable, fontsize=16)
    matplotlib.pyplot.xlabel(r'$%s$' % variable, fontsize=16)
#    matplotlib.pyplot.title(png)
    if round(xr[1]/xr[0]) == 1e1:
        x_text = 1.27
    elif round(xr[1]/xr[0]) == 1e2:
        x_text = 1.6
    elif round(xr[1]/xr[0]) == 1e3:
        x_text = 2
    elif round(xr[1]/xr[0]) == 1e4:
        x_text = 2.5
    elif round(xr[1]/xr[0]) == 1e5:
        x_text = 3
    elif round(xr[1]/xr[0]) == 1e6:
        x_text = 3.9
    elif round(xr[1]/xr[0]) == 1e7:
        x_text = 5
    else:
        x_text = 2
    if round(yr[1]/yr[0]) == 1e3:
        y_text = (4.3, 2.9, 1.7)
    elif round(yr[1]/yr[0]) == 1e4:
        y_text = (7, 4, 2)
    elif round(yr[1]/yr[0]) == 1e5:
        y_text = (11, 5.7, 2.4)
    elif round(yr[1]/yr[0]) == 1e6:
        y_text = (19, 8, 3)
    else:
        y_text = (7, 4, 2)
    matplotlib.pyplot.text(x_text*xr[0], y_text[0]*yr[0], r'$\alpha=%g$' % alpha, fontsize=14)
    matplotlib.pyplot.text(x_text*xr[0], y_text[1]*yr[0], r'$%s_{min}=%g$' % (variable, xmin), fontsize=14)
    if p_value is not None:
        matplotlib.pyplot.text(x_text*xr[0], y_text[2]*yr[0], r'$p$-value $=%g$' % p_value, fontsize=14)
    matplotlib.pyplot.savefig(png)

    return h
