# KernelDensityEstimators

*If you found this useful and would like to buy me a coffee, please consider donating to the wallet:*

☕ 1PGmwg3hQN78XTg8PNydbPWpeiupeQ15dj ☕

The KDE object in this project allows you to estimate the distribution of numerical data with user-defined granularity by introducing the free parameter `cuts`. This parameter allows for control over the order of estimation.

The methodology for construction is outlined in the notebook `kde.ipynb` and the neccessary code for computation is given in `KernelDensityEstimator.py`.

Current KDE options include:
- Gaussian (defualt, and most robust estimator)
- Cauchy
- Poisson (not very useful but has its uses; can be used for skew-left-decreasing data on the interval `[0,N]`)

<img src=comparison.png>

## Some Quantitative Comments:

If estimating using Gaussian, I recommend using something like Sturgis' Rule (`cuts = 1+3.3log10(n)`, where `n = len(data)`) for determining the number of cuts to use. Obviously, increasing the number of cuts too high will result in a gross overfitting of the estimation. 

If estimating using Cauchy, I find that increasing the number of cuts by around a factor of 2 helps the appearance of the estimation. Using something like Rice's Rule (`cuts = 2n^(1/3)`, where `n = len(data)`) is beneficial here.

If estimating using Poisson, ensure that 1) The support is `[0,N]` for some large integer `N`, 2) The data is roughly monotone decreasing. If it can be helped, try to additionally use integer valued data (this is not neccessary however).


## In Construction:

Including a method which will return the analytic expression of an estimation.
