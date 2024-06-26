% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/compute-AEL-functions.R
\name{compute_AEL}
\alias{compute_AEL}
\title{Compute the Adjusted Empirical Likelihood}
\usage{
compute_AEL(th, h, lam0, a, z, iters, useR_forz, returnH)
}
\arguments{
\item{th}{Vector or scalar theta}

\item{h}{User-defined function, outputs array}

\item{lam0}{Initial vector for lambda}

\item{a}{Scalar constant}

\item{z}{n-1 by d matrix}

\item{iters}{Number of iterations using Newton-Raphson for estimation of lambda (default: 500)}

\item{useR_forz}{Bool whether to calculate the function first in R (True) or call the function in C (False) (default: True)}

\item{returnH}{Whether to return calculated values of h, H matrix and lambda}
}
\value{
A numeric value for the Adjusted Empirical Likelihood function
computed evaluated at a given theta value
}
\description{
Evaluates the AEL for a given data set, moment conditions and parameter values
}
\examples{
# Generate toy variables
set.seed(1)
x     <- runif(30, min = -5, max = 5)
elip  <- rnorm(30, mean = 0, sd = 1)
y     <- 0.75 - x + elip

# Set initial values for AEL computation
lam0  <- matrix(c(0,0), nrow = 2)
th    <- matrix(c(0.8277, -1.0050), nrow = 2)
a     <- 0.00001
iters <- 10

# Define Dataset and h-function
z <- cbind(x, y)
h <- function(z, th) {
    xi      <- z[1]
    yi      <- z[2]
    h_zith  <- c(yi - th[1] - th[2] * xi, xi*(yi - th[1] - th[2] * xi))
    matrix(h_zith, nrow = 2)
}
ansAELRcpp <- compute_AEL(th, h, lam0, a, z, iters, useR_forz = TRUE)
}
\references{
Yu, W., & Bondell, H. D. (2023). Variational Bayes for Fast and
Accurate Empirical Likelihood Inference. Journal of the American Statistical
Association, 1–13. \doi{doi:10.1080/01621459.2023.2169701}
}
\author{
Wei Chang Yu, Jeremy Lim
}
