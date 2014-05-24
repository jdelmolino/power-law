# May 2, 2010

import numpy
import scipy.special

def plfit0(x, **kwargs):
    """
% PLFIT0 tries to fit a power-law distributional model to discrete data.
% 
%    PLFIT0(x) estimates xmin and alpha according to the goodness-of-fit
%    based method described in Clauset, Shalizi, Newman (2007). x is a 
%    vector of observations of some quantity to which we wish to fit the 
%    power-law distribution p(x) ~ x**-alpha for x >= xmin.
%    PLFIT0 only works if x is composed of integer values.
%   
%    The fitting procedure works as follows:
%    1) For each possible choice of xmin, we estimate alpha via the
%       approximate expression, and calculate the Kolmogorov-Smirnov
%       goodness-of-fit (gof) statistic D.
%    2) We then select as our estimate of xmin, the value that gives the
%       minimum value D over all values of xmin.
%
%    Note that this procedure gives no estimate of the uncertainty of the 
%    fitted parameters, nor of the validity of the fit.
%
%    Example:
%       xd = numpy.floor((xmin-0.5)*(1-numpy.random.random(10000))**(-1/(alpha-1))+0.5)
%       alpha, xmin = plfit0.plfit0(xd)
%
%    The output 'alpha' is the maximum likelihood estimate of the scaling
%    exponent, 'xmin' is the estimate of the lower bound of the power-law
%    behavior.
%
%    See also PLFIT, PLVAR, PLPVA
%
% Notes:
%    
% 1. PLFIT can be told to limit the range of values considered as estimates
%    for xmin in two ways. First, it can be instructed to sample these
%    possible values like so,
%    
%       a, xmin = plfit0.plfit0(x, sample=100)
%    
%    which uses 100 uniformly distributed values on the sorted list of
%    unique values in the data set. Alternatively, it can simply omit all
%    candidates above a hard limit, like so
%    
%       a, xmin = plfit0.plfit0(x, limit=3.4)
%    
%    It rounds the limit to the nearest integer.
% 
    """

    x = numpy.array(x)
    x = x[x>0.5]
    
    # discrete method
    xmins = numpy.unique(x)
    xmins = xmins[:-1]
    try:
        xmins = xmins[numpy.array(numpy.unique(numpy.round(numpy.linspace(1, len(xmins), kwargs['sample']) - 1)), dtype='int32')]
    except KeyError:
        pass
    try:
        limit = round(kwargs['limit'])
        xmins = xmins[xmins <= limit]
    except KeyError:
        pass
    if len(xmins) == 0:
        print '(PLFIT0) Error: x must contain at least two unique values.'
        alpha = numpy.nan
        xmin = x[0]
    else:
        xmax = max(x)
        dat  = numpy.zeros(len(xmins))
        z    = numpy.sort(x)
        for exmin, xmin in enumerate(xmins):
            z = z[z >= xmin]
            n = float(len(z))
            # estimate alpha via the approximate expression
            a  = 1 + n / sum(numpy.log(z/(xmin-0.5))) # (3.7) (B.17)  
            za = scipy.special.zeta(a, xmin) # (2.5)
            # compute KS statistic
            cdi = numpy.cumsum(numpy.histogram(z, numpy.arange(xmin, xmax+2))[0] / n) # S(x)
            fit = numpy.cumsum((numpy.arange(xmin, xmax+1)**-a) / za) # P(x)
            dat[exmin] = max(abs( cdi - fit )) # (3.9)
        # select the index for the minimum value of D
        D, I  = dat.min(0), dat.argmin(0)
        xmin  = xmins[I]
        z     = numpy.sort(x)
        z     = z[z >= xmin]
        n     = float(len(z))
        alpha = 1 + n / sum(numpy.log(z/(xmin-0.5))) # (3.7) (B.17)

    return alpha, xmin
