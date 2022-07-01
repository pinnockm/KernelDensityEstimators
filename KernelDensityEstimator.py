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

def poisson(x, mu:float, continuous:bool=True):
    """Return the Poisson distribution over the support x.
    PRECONDITION: min(x), mu >= 0.
    """
    if (mu < 0) | (min(x) < 0):
        raise ValueError("Min(x) and mu must be non-negative.")

    from scipy.special import gamma as Gamma
    from scipy.integrate import quad
    # integrate to find appropriate normalization constant
    scale = 1/quad(lambda x: mu**x/Gamma(1+x)*np.exp(-mu), a=0, b=200)[0]
    if continuous:
        return scale * mu**x/Gamma(1+x) * np.exp(-mu)
    return mu**x/Gamma(1+x) * np.exp(-mu)

# =========================================================================== #

class KDE:

    def __init__(self, arr, cuts:int = 10) -> None:
        if cuts < 2:
            raise ValueError("k must be greater than 1.")
        
        self.k = cuts
        self.arr = arr
        self.stats = self.__stats()
        pass
    
    def __stats(self):
        from scipy.stats import skew
        return {
            "Mean": np.mean(self.arr),
            "Variance": np.var(self.arr),
            "Skewness": skew(self.arr),
            "Quartiles": [np.quantile(self.arr, q/4) for q in range(5)]
        }

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
        """Returns a tuple of pdf support and density function
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
        x = np.linspace(start=0.95*self.subintervals()[0][0],
                        stop=1.05*self.subintervals()[-1][1],
                        num=1000)
        
        pdf = 0
        # Gaussian RV
        if dist == 'gaussian':
            for alpha,mu in zip(alpha_i, mu_i):
                # weighted sum of gaussians
                # centered at midpoints of the intervals
                # variance equal to half-width of the intervals
                pdf += alpha * gaussian(x, mu, sigma)
            return (x, pdf/len(self.arr))   # tuple of support and pdf
        
        # Cauchy RV
        elif dist == 'cauchy':
            print("Suggestion: if estimating with Cauchy, use at least twice as many cuts as you would for Gauss.")
            for alpha,mu in zip(alpha_i, mu_i):
                # weighted sum of Cauchy random variables
                # centered at midpoints of the intervals
                # variance equal to the full width of the intervals
                pdf += alpha * cauchy(x, mu, 2*sigma)
            return (x, pdf/len(self.arr))   # tuple of support and pdf

        # Poisson RV
        else:
            from warnings import warn
            warn(
                "Poisson requires a support [0,inf). At the least, ensure that x = [0,N] for large N.",
                category=UserWarning)
            
            print("Note: Poisson should only be used for skew-left integer data, decreasing on a support [0,N].")
            
            for alpha,mu in zip(alpha_i, mu_i):
                # weighted sum of Poisson random variables
                # centered at midpoints of the intervals
                pdf += alpha * poisson(x, mu, continuous)
            return (x, pdf/len(self.arr))   # tuple of support and pdf

    def analytic_expression(self, dist:str = 'gaussian', continuous:bool=True):
        """Takes in a distribution type and returns the analytic expression of
        the generated Kernel Density Estimate. Possible inputs for dist
        include: 'gaussian', 'cauchy', 'poisson'."""

        if dist not in ['gaussian', 'cauchy', 'gamma', 'poisson']:
            raise ValueError("Distribution needs to be one of ['gaussian','cauchy','gamma','poisson'].")

        # coefficients for KDE
        alpha_i = list(self.indicator_coefs().values())
        # locations of the center of the probability mass
        mu_i = self.midpoints()
        # dispersion of the distribution
        width = self.subintervals()[0][1] - self.subintervals()[0][0]
        sigma = 0.5 * width
        
        pdf_exp = []
        # Gaussian RV
        if dist == 'gaussian':
            for alpha,mu in zip(alpha_i, mu_i):
                # weighted sum of gaussians
                # centered at midpoints of the intervals
                # variance equal to half-width of the intervals
                pdf_exp.append(f"{alpha}*N(x;{mu},{sigma})")
            return f"[{' + '.join(pdf_exp)}]/{len(self.arr)}"
        
        # Cauchy RV
        elif dist == 'cauchy':
            for alpha,mu in zip(alpha_i, mu_i):
                # weighted sum of Cauchy random variables
                # centered at midpoints of the intervals
                # variance equal to the full width of the intervals
                pdf_exp.append(f"{alpha}*Cauchy(x; {mu}, {2*sigma})")
            return f"[{' + '.join(pdf_exp)}]/{len(self.arr)}"

        # Poisson RV
        else:
            from warnings import warn
            warn(
                "Poisson requires a support [0,inf). At the least, ensure that x = [0,N] for large N.",
                category=UserWarning)

            for alpha,mu in zip(alpha_i, mu_i):
                # weighted sum of Poisson random variables
                # centered at midpoints of the intervals
                pdf_exp.append(f"{alpha}*Poisson(x; {mu})")
            return f"[{' + '.join(pdf_exp)}]/{len(self.arr)}"