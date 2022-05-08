import numpy as np

###############################################################################
##                          Probability Densities                            ##
###############################################################################

def gaussian(x, mu:float, var:float):
    """Return the Gaussian distribution over support x."""
    return np.exp(-((x-mu)/(var*np.sqrt(2)))**2)/(var*np.sqrt(2*np.pi))

def cauchy(x, loc:float, scale:float):
    """Return the Cauchy distribution over the support x.
    PRECONDITION: scale > 0.
    """
    if scale <= 0:
        raise ValueError("Scale must be greater than 0.")
    return (np.pi*scale* (1 + ((x-loc)/scale)**2 ))**(-1)

# helper function for poisson
def continuum_approx(mu:float) -> float:
    """Returns a float which approximately scales the continuous integral of
    the poisson distribution to unity. Coefficients to an odd-degree Taylor
    expansion T(x)=\sum^5_{i=0}c_ix^i found by multilinear regression."""
    # regression on the ratio of \int^\infty_0 f(x)dx to \sum^\infty_{i=0}f(i)
    # coefficients to T_5(x) civen by c_i
    # odd-degree to suit the end behaviour of arctan and the residuals
    ci = [0.369865360931, 4.11154855117, -4.51929684625, 5.65238457124,
          -2.26404038444, 0.386793184544]
    mui = [pow(mu,i) for i in range(len(ci))]
    return (np.pi/2) * 1/np.arctan(np.dot(ci,mui))

def poisson(x, mu:float, continuous:bool=True):
    """Return the Poisson distribution over the support x.
    PRECONDITION: min(x), mu >= 0.
    """
    if (mu < 0) | (min(x) < 0):
        raise ValueError("Min(x) and mu must be non-negative.")

    from scipy.special import gamma as Gamma
    if continuous:
        return continuum_approx(mu) * mu**x/Gamma(1+x) * np.exp(-mu)
    return mu**x/Gamma(1+x) * np.exp(-mu)

# =========================================================================== #

class KDE:

    def __init__(self, arr, cuts:int = 100) -> None:
        if cuts < 2:
            raise ValueError("k must be greater than 1.")
        
        self.k = cuts
        self.arr = arr
        pass

    def subintervals(self) -> list:
        """Takes a discrete array of numerical values and an integral
        number of cuts k (default 100). Returns a list of interval endpoints on
        the real line."""
        
        low, high = min(self.arr), max(self.arr)
        si = np.linspace(low, high, self.k)
        return [(si[i],si[i+1]) for i in range(len(si)-1)]

    def indicator_coefs(self) -> dict:
        """Return a count of values from arr which fall between the bounds
        of the subintervals. Where subintervals is an array-like of the
        form subintervals_arr()."""

        alpha = dict().fromkeys(range(len(self.subintervals())), 0)
        for n,(i,j) in enumerate(self.subintervals()):
            for x in self.arr:
                if i <= x <= j:
                    alpha[n] += 1    # alpha_n
        return alpha

    def midpoints(self) -> list:
        """Takes in an array-like of the form [(x,y)]. Returns a list of
        midpoints for the subintervals."""

        midpts = []
        for (i,j) in self.subintervals():
            midpts.append((i+j)/2)
        return midpts

    def density(self, dist:str ='gaussian', continuous:bool=True) -> tuple:
        """Takes in an array-like of numerical data and an array of
        subintervals. Returns a tuple of pdf support and density function
        estimating the histogram of arr. Possible inputs for dist include:
        'gaussian', 'cauchy', 'poisson'.

        Continuous parameter is for the backwards compatability with poisson.

        Note: poisson should only be used for skew-left decreasing data.
        """

        if dist not in ['gaussian', 'cauchy', 'gamma', 'poisson']:
            raise ValueError("Distribution needs to be one of ['gaussian','cauchy','gamma','poisson'].")

        # coefficients for KDE
        alpha_i = list(self.indicator_coefs().values())
        # locations of the center of the probability mass
        mu_i = self.midpoints()
        # dispersion of the distribution
        width = self.subintervals()[0][1] - self.subintervals()[0][0]
        sigma = 0.5 * width
        
        # support x
        x = np.linspace(start=0.9*self.subintervals()[0][0],
                        stop=1.1*self.subintervals()[-1][1],
                        num=1000)
        
        pdf = 0
        if dist == 'gaussian':
            for alpha,mu in zip(alpha_i, mu_i):
                pdf += alpha * gaussian(x, mu, sigma)
            return (x, pdf/len(self.arr))   # tuple of support and pdf
        
        elif dist == 'cauchy':
            print("Suggestion: if estimating with Cauchy, use at least twice as many cuts as you would for Gauss.")
            for alpha,mu in zip(alpha_i, mu_i):
                pdf += alpha * cauchy(x, mu, 2*sigma)
            return (x, pdf/len(self.arr))   # tuple of support and pdf

        else:
            from warnings import warn
            warn(
                "Poisson requires a support [0,inf). At the least, ensure that x = [0,N] for large N.",
                category=UserWarning)
            
            print("Note: Poisson should only be used for skew-left integer data, decreasing on a support [0,N].")
            
            for alpha,mu in zip(alpha_i, mu_i):
                pdf += alpha * poisson(x, mu, continuous)
            return (x, pdf/len(self.arr))   # tuple of support and pdf
