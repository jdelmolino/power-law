# August 12, 2009

import numpy

def randht(option, n, xmin, *param):
    """
% RANDHT generates n observations distributed as some continous heavy-
% tailed distribution. Options are power law, log-normal, stretched 
% exponential, power law with cutoff, and exponential. Can specify lower 
% cutoff, if desired.
% 
%    Example:
%       x = randht.randht('powerlaw', 10000, 1, alpha)        # power law 
%       x = randht.randht('cutoff', 10000, 1, alpha, lamda)   # power law with cutoff
%       x = randht.randht('exponential', 10000, 1, lamda)     # exponential
%       x = randht.randht('stretched', 10000, 1, lamda, beta) # stretched exponential
%       x = randht.randht('lognormal', 10000, 1, mu, sigma)   # log-normal
%
%    See also PLFIT, PLVAR, PLPVA
%
    """

    if n < 1:
        n = 1e4
        print '(RANDHT) Error: invalid "n" argument; using default (%d)' % n
    if xmin < 1:
        xmin = 1
        print '(RANDHT) Error: invalid "xmin" argument; using default (%d)' % xmin

    # parse command-line parameters

    if option == 'powerlaw':
        alpha = param[0]
        x = xmin*(1 - numpy.random.random(n))**(-1/(alpha - 1))

    elif option == 'cutoff':
        alpha = param[0]
        lamda = param[1]
        x = numpy.arange(0, dtype='float64')
        y = xmin - (1/lamda)*numpy.log(1 - numpy.random.random(10*n))
        while True:
            y = y[numpy.random.random(10*n) < (y/xmin)**-alpha]
            x = numpy.concatenate((x, y))
            q = len(x) - n
            if (q == 0):
                break
            if (q > 0):
                r = numpy.random.permutation(len(x))
                x = x.take(r[:q])
                break
            if (q < 0):
                y = xmin - (1/lamda)*numpy.log(1 - numpy.random.random(10*n))
 
    elif option == 'exponential':
        lamda = param[0]
        x = xmin - (1/lamda)*numpy.log(1 - numpy.random.random(n))

    elif option == 'stretched':
        lamda = param[0]
        beta = param[1]
        x = (xmin**beta - (1/lamda)*numpy.log(1 - numpy.random.random(n)))**(1/beta)

    elif option == 'lognormal':
        mu = param[0]
        sigma = param[1]
        y = numpy.exp(mu + sigma*numpy.random.standard_normal(10*n))
        while True:
            y = y[y >= xmin]
            q = len(y) - n
            if (q == 0):
                break
            if (q > 0):
                r = numpy.random.permutation(len(y))
                y = y.take(r[:q])
                break
            if (q < 0):
                y = numpy.concatenate((y, numpy.exp(mu + sigma*numpy.random.standard_normal(10*n))))
        x = y

    return x
