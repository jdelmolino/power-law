# May 7, 2010

import numpy
import scipy.special

def plfit(x, vec=numpy.arange(1.50, 3.51, 0.01), nosmall=False, finite=False, **kwargs):
    """
% PLFIT fits a power-law distributional model to data.
% 
%    PLFIT(x) estimates xmin and alpha according to the goodness-of-fit
%    based method described in Clauset, Shalizi, Newman (2007). x is a 
%    vector of observations of some quantity to which we wish to fit the 
%    power-law distribution p(x) ~ x**-alpha for x >= xmin.
%    PLFIT automatically detects whether x is composed of real or integer
%    values, and applies the appropriate method. For discrete data, if
%    min(x) > 1000, PLFIT uses the continuous approximation, which is 
%    a reliable in this regime.
%   
%    The fitting procedure works as follows:
%    1) For each possible choice of xmin, we estimate alpha via the 
%       method of maximum likelihood, and calculate the Kolmogorov-Smirnov
%       goodness-of-fit (gof) statistic D.
%    2) We then select as our estimate of xmin, the value that gives the
%       minimum value D over all values of xmin.
%
%    Note that this procedure gives no estimate of the uncertainty of the 
%    fitted parameters, nor of the validity of the fit.
%
%    Example:
%       xc = xmin*(1-numpy.random.random(10000))**(-1/(2.5-1))
%       alpha, xmin, L = plfit.plfit(xc)
%       xd = numpy.floor((xmin-0.5)*(1-numpy.random.random(10000))**(-1/(alpha-1))+0.5)
%       alpha, xmin, L = plfit.plfit(xd)
%
%    The output 'alpha' is the maximum likelihood estimate of the scaling
%    exponent, 'xmin' is the estimate of the lower bound of the power-law
%    behavior, and L is the logarithm of the likelihood of the data x >= xmin
%    under the fitted power law.
%
%    See also PLVAR, PLPVA
%
% Notes:
% 
% 1. In order to implement the integer-based methods, the numeric
%    maximization of the log-likelihood function was used. This requires
%    that we specify the range of scaling parameters considered. We set
%    this range to be numpy.arange(1.50, 3.51, 0.01) by default. This vector
%    can be set by the user like so,
%    
%       a, xmin, L = plfit.plfit(x, vec=numpy.arange(1.001, 5.002, 0.001))
%    
% 2. PLFIT can be told to limit the range of values considered as estimates
%    for xmin in two ways. First, it can be instructed to sample these
%    possible values like so,
%    
%       a, xmin, L = plfit.plfit(x, sample=100)
%    
%    which uses 100 uniformly distributed values on the sorted list of
%    unique values in the data set. Alternatively, it can simply omit all
%    candidates above a hard limit, like so
%    
%       a, xmin, L = plfit.plfit(x, limit=3.4)
%    
%    In the case of discrete data, it rounds the limit to the nearest
%    integer.
% 
% 3. When the input sample size is small (e.g., < 100), the continuous 
%    estimator is slightly biased (toward larger values of alpha). To
%    explicitly use an experimental finite-size correction, call PLFIT like
%    so
%    
%       a, xmin, L = plfit.plfit(x, finite=True)
%    
%    which does a small-size correction to alpha.
%
% 4. For continuous data, PLFIT can return erroneously large estimates of 
%    alpha when xmin is so large that the number of obs x >= xmin is very 
%    small. To prevent this, we can truncate the search over xmin values 
%    before the finite-size bias becomes significant by calling PLFIT as
%    
%       a, xmin, L = plfit.plfit(x, nosmall=True)
%    
%    which skips values xmin with finite size bias > 0.1.
%
    """

    x = numpy.array(x)
    x = x[x>0]

    xmins = numpy.unique(x)
    xmins = xmins[:-1]
    try:
        xmins = xmins[numpy.array(numpy.unique(numpy.round(numpy.linspace(1, len(xmins), kwargs['sample']) - 1)), dtype='int32')]
    except KeyError:
        pass

    # continuous method
    if (min(x) > 1000 and len(x) > 100) or sum(x-numpy.floor(x)):
        try:
            xmins = xmins[xmins <= kwargs['limit']]
        except KeyError:
            pass
        dat = numpy.array([])
        z   = numpy.sort(x)
        for xmin in xmins:
            z = z[z >= xmin]
            n = float(len(z))
            # estimate alpha using direct MLE
            a = n / sum(numpy.log(z/xmin))
            if nosmall:
                if (a-1)/numpy.sqrt(n) > 0.1:
                    break
            # compute KS statistic
            cx = numpy.arange(n) / n
            cf = 1 - (xmin/z)**a
            dat = numpy.r_[dat, max(abs( cf - cx ))]
        D     = min(dat)
        xmin  = xmins[(dat<=D).nonzero()[0][0]]
        z     = numpy.sort(x)
        z     = z[z >= xmin]
        n     = float(len(z))
        alpha = 1 + n / sum(numpy.log(z/xmin))
        if finite:
            alpha = alpha*(n-1)/n + 1/n # finite-size correction
        else:
            if n < 50:
                print '(PLFIT) Warning: finite-size bias may be present (n = %d).' % n
        L = n * numpy.log((alpha-1)/xmin) - alpha * sum(numpy.log(z/xmin))

    # discrete method
    else:
        try:
            limit = round(kwargs['limit'])
            xmins = xmins[xmins <= limit]
        except KeyError:
            pass
        if len(xmins) == 0:
            print '(PLFIT) Error: x must contain at least two unique values.'
            alpha = numpy.nan
            xmin = x[0]
            L = numpy.nan
        else:
            xmax   = max(x)
            dat    = numpy.zeros((len(xmins), 2))
            z      = numpy.sort(x)
            fcatch = 0
            for xm, xmin in enumerate(xmins):
                zvec  = scipy.special.zeta(vec, xmin) # (2.5)
                z     = z[z >= xmin]
                n     = float(len(z))
                slogz = sum(numpy.log(z))
                # estimate alpha via direct maximization of likelihood function
                if fcatch == 0:
                    try:
                        # vectorized version of numerical calculation
                        L = - n*numpy.log(zvec)- vec*slogz # (3.5) (B.8)
                    except:
                        # catch: force loop to default to iterative version for
                        # remainder of the search
                        print "except"
                        fcatch = 1
                if fcatch == 1:
                    # force iterative calculation (more memory efficient, but 
                    # can be slower)
                    L = - numpy.inf * numpy.ones(len(vec))
                    for k in xrange(len(vec)):
                        L[k] = - n*numpy.log(zvec[k]) - vec[k]*slogz # (3.5) (B.8)
                Y, I = L.max(0), L.argmax(0)
                # compute KS statistic
                cdi = numpy.cumsum(numpy.histogram(z, numpy.arange(xmin, xmax+2))[0] / n) # S(x)
                fit = numpy.cumsum((numpy.arange(xmin, xmax+1)**-vec[I]) / zvec[I]) # P(x)
                dat[xm] = [max(abs( cdi - fit )), vec[I]] # (3.9)
            # select the index for the minimum value of D
            D, I  = dat[:, 0].min(0), dat[:, 0].argmin(0)
            xmin  = xmins[I]
            n     = sum(x[x>=xmin])
            alpha = dat[I, 1]
            if finite:
                alpha = alpha*(n-1)/n + 1/n # finite-size correction
            else:
                if n < 50:
                    print '(PLFIT) Warning: finite-size bias may be present (n = %d).' % n
            L = - alpha*sum(numpy.log(z)) - n*numpy.log(scipy.special.zeta(alpha, xmin) - scipy.special.zeta(alpha, xmax+1))

    return alpha, xmin, L

def plfit_xmax(x, vec=numpy.arange(1.50, 3.51, 0.01), finite=False, sample=0.1):
    # discrete method
    x = numpy.array(x)
    N = float(len(x))
    xunique = numpy.unique(x)

    if len(xunique) == 1:
        print '(PLFIT) Error: x must contain at least two distinct values.'
        alpha = numpy.nan
        xmin = xunique[0]
        xmax = xunique[0]
        L = numpy.nan
    else:
        xmins = xunique[:-1]
        dat   = numpy.inf * numpy.ones((len(xmins)*(len(xmins)+1)/2, 2))
        xsort = numpy.sort(x)
        for exmin, xmin in enumerate(xmins):
            xmaxs = xunique[exmin+1:]
            for exmax, xmax in enumerate(xmaxs):
                zvec = scipy.special.zeta(vec, xmin) - scipy.special.zeta(vec, xmax+1) # (2.5)
                z    = xsort[xsort >= xmin]
                z    = z[z <= xmax]
                n    = float(len(z))
                if n > sample*N: # 10% by default
                    slogz = sum(numpy.log(z))
                    # vectorized version of numerical calculation
                    L    = - n*numpy.log(zvec) - vec*slogz # (3.5) (B.8)
                    Y, I = L.max(), L.argmax()
                    # compute KS statistic
                    cdi = numpy.cumsum(numpy.histogram(z, numpy.arange(xmin, xmax+2))[0] / n) # S(x)
                    fit = numpy.cumsum((numpy.arange(xmin, xmax+1)**-vec[I]) / zvec[I]) # P(x)
                    dat[exmin*(2*len(xmins)-exmin+1)/2 + exmax] = [max(abs( cdi - fit )), vec[I]] # (3.9)
        # select the indexes for the minimum value of D
        D, I       = dat[:, 0].min(0), dat[:, 0].argmin(0)
        for exmin, xmin in enumerate(xmins):
            xmaxs = xunique[exmin+1:]
            for exmax, xmax in enumerate(xmaxs):
                if I == exmin*(2*len(xmins)-exmin+1)/2 + exmax:
                    break
            if I == exmin*(2*len(xmins)-exmin+1)/2 + exmax:
                break
        z     = x[x>=xmin]
        z     = z[z<=xmax]
        n     = sum(z)
        alpha = dat[I][1]
        if finite:
            alpha = alpha*(n-1)/n + 1/n # finite-size correction
        else:
            if n < 50:
                print '(PLFIT) Warning: finite-size bias may be present.'
        L = - alpha*sum(numpy.log(z)) - n*numpy.log(scipy.special.zeta(alpha, xmin) - scipy.special.zeta(alpha, xmax+1))
        
    return alpha, xmin, xmax, L
