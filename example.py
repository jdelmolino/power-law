"""
Example to use plfit0, plfit, plpva, plplot (discrete data)

Calls plfit0.plfit0, plfit.plfit, plpva.plpva and plplot.plplot in order

Usage: python example.py [options]

Options:
    -h, --help   show this help and exit
"""

import getopt
import numpy
import sys

import plfit0
import plfit
import plpva
import plplot

def usage():
    print __doc__

def power_law(x, variable, subject, radius = 0.5, number_of_sets = 100):
    print "- Fitting power law to empirical data: %s" % variable
    if sum(x-numpy.floor(x)):
        print "  CONTINUOUS"
        alpha_range = None
    else:
        alpha, xmin = plfit0.plfit0(x)
        print "  DISCRETE"
        print "  Approximate estimator for the scaling parameter of the discrete power law:"
        print "  * Scaling parameter: alpha %g" % alpha
        print "  * Lower bound: xmin %g" % xmin
        alpha_range = numpy.arange(round(alpha)-radius, round(alpha)+radius, 0.001)
        alpha_range = alpha_range[alpha_range > 1] # distributions with alpha <=1 are not normalizable
    alpha, xmin, L = plfit.plfit(x, vec = alpha_range, nosmall = False, finite = True)
    print "  Numerical maximization of the logarithm of the likelihood function L:"
    print "  * Scaling parameter: alpha %g" % alpha
    try:
        if alpha == min(alpha_range) or alpha == max(alpha_range):
            print "    WARNING alpha_range"
    except TypeError:
        pass
    print "  * Lower bound: xmin %g" % xmin
    print "  * Logarithm of the likelihood function: L %g" % L
    p, gof = plpva.plpva(x, xmin, vec=alpha_range, reps=number_of_sets, quiet=True)
    print "  Generation of %d power-law distributed synthetic data sets:" % number_of_sets
    print "  * Fraction of data sets with worse KS statistic than the empirical data: p-value %g" % p
    print "  * KS statistic of the empirical data: D %g" % gof
    png = "plplot_"+subject
    plplot.plplot(x, xmin, alpha, variable, p, png)

def main(argv):
    """Gets command arguments from the shell: 'help'
    """
    try:
        opts, args = getopt.getopt(argv, "h", ["help"])
    except getopt.GetoptError:
        usage()
        sys.exit(1)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit(1)

    # power-law
    x = numpy.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4], dtype=float) # sample data
    variable = "n"
    subject = "example"
    radius = 0.5
    number_of_sets = 100
    power_law(x, variable, subject, radius, number_of_sets)
    sys.exit(0)

if __name__ == '__main__':
    main(sys.argv[1:])
