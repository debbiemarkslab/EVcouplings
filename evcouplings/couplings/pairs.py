"""
Functions for handling evolutionary couplings data.

.. todo::

    1. clean up
    2. add Pompom score
    3. add mapping tools (multidomain, complexes)
    4. ECs to matrix
    5. APC on subsets of positions (e.g. for complexes)

Authors:
  Thomas A. Hopf
  Agnes Toth-Petroczy (original mixture model code)
  John Ingraham (skew normal mixture model)
"""

from math import ceil
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as op
from scipy import stats


def read_raw_ec_file(filename, sort=True, score="cn"):
    """
    Read a raw EC file (e.g. from plmc) and sort
    by scores

    Parameters
    ----------
    filename : str
        File containing evolutionary couplings
    sort : bool, optional (default: True)
        If True, sort pairs by coupling score in
        descending order
    score_column : str, optional (default: True)
        Score column to be used for sorting

    Returns
    -------
    ecs : pd.DataFrame
        Table of evolutionary couplings

    """
    ecs = pd.read_csv(
        filename, sep=" ",
        names=["i", "A_i", "j", "A_j", "fn", "cn"]
    )

    if sort:
        ecs = ecs.sort_values(
            by=score, ascending=False
        )

    return ecs


def enrichment(ecs, num_pairs=1.0, score="cn", min_seqdist=6):
    """
    Calculate EC "enrichment" as first described in
    Hopf et al., Cell, 2012.

    .. todo::

        Make this handle segments if they are in EC table

    Parameters
    ----------
    ecs : pd.DataFrame
        Dataframe containing couplings
    num_pairs : int or float, optional (default: 1.0)
        Number of ECs to use for enrichment calculation.
        - If float, will be interpreted as fraction of the
        length of the sequence (e.g. 1.0*L)
        - If int, will be interpreted as
        absolute number of pairs
    score : str, optional (default: cn)
        Pair coupling score used for calculation
    min_seqdist : int, optional (default: 6)
        Minimum sequence distance of couplings that will
        be included in the calculation

    Returns
    -------
    enrichment_table : pd.DataFrame
        Sorted table with enrichment values for each
        position in the sequence
    """
    # determine how many positions ECs are over
    pos = set(ecs.i.unique()) | set(ecs.j.unique())
    num_pos = len(pos)

    # calculate absolute number of pairs if
    # fraction of length is given
    if isinstance(num_pairs, float):
        num_pairs = int(ceil(num_pairs * num_pos))

    # get longrange ECs and sort by score
    sorted_ecs = ecs.query(
        "abs(i-j) >= {}".format(min_seqdist)
    ).sort_values(
        by=score, ascending=False
    )

    # select top EC pairs
    top_ecs = sorted_ecs.iloc[0:num_pairs]

    # stack dataframe so it contains each
    # EC twice as forward and backward pairs
    # (i, j) and (j, i)
    flipped = top_ecs.rename(
        columns={"i": "j", "j": "i", "A_i": "A_j", "A_j": "A_i"}
    )

    stacked_ecs = top_ecs.append(flipped)

    # now sum cumulative strength of EC for each position
    ec_sums = pd.DataFrame(
        stacked_ecs.groupby(["i", "A_i"]).sum()
    )

    # average EC strength for top ECs
    avg_degree = top_ecs.loc[:, score].sum() / len(top_ecs)

    # "enrichment" is ratio how much EC strength on
    # an individual position exceeds average strength in top
    ec_sums.loc[:, "enrichment"] = ec_sums.loc[:, score] / avg_degree

    e = ec_sums.reset_index().loc[:, ["i", "A_i", "enrichment"]]
    return e.sort_values(by="enrichment", ascending=False)


class LegacyScoreMixtureModel:
    """
    Assign to each EC score the probability of being in the
    lognormal tail of a normal-lognormal mixture model.

    .. note::
        this is the original version of the score mixture model
        with a normal distribution noise component, this has been
        superseded by a model using a skew normal distribution
    """
    def __init__(self, x, clamp_mu=False, max_fun=10000, max_iter=1000):
        """
        Mixture model of evolutionary coupling scores to
        determine significant scores that are in high-scoring,
        positive tail of distribution.

        Parameters
        ----------
        x : np.array (or list-like)
            EC scores from which to infer the mixture model
        clamp_mu : bool, optional (default: False)
            Fix mean of Gaussian component to 0 instead of
            fitting it based on data
        max_fun : int
            Maximum number of function evaluations
        max_iter : int
            Maximum number of iterations
        """
        x = np.array(x)

        # Infer parameters of mixture model
        self.params = self._learn_params(x, clamp_mu, max_fun, max_iter)

    @classmethod
    def _learn_params(cls, x, clamp_mu, max_fun, max_iter):
        """
        Infer parameters of mixture model.

        Parameters
        ----------
        x : np.array (or list-like)
            EC scores from which to infer the mixture model
        clamp_mu : bool, optional (default: False)
            Fix mean of Gaussian component to 0 instead of
            fitting it based on data
        max_fun : int
            Maximum number of function evaluations
        max_iter : int
            Maximum number of iterations

        Returns
        -------
        mu : float
            Mean of normal distribution
        sigma : float
            Standard deviation of normal distribution
        q : float
            Relative weights of each distribution
        logmu : float
            Mean of lognormal distribution
        logsigma : float
            Standard deviation of lognormal distribution
        """
        # Initial starting parameters for normal
        # and lognormal distributions
        # q: relative contribution of each distribtuion
        mu = 0
        sigma = np.std(x)
        q = 1
        logsigma = 0.4
        logmu = np.percentile(x, 75) - logsigma**2 / 2
        param = np.array([mu, sigma, q, logmu, logsigma])

        # Target function for minimization
        def target_func(params):
            if clamp_mu:
                params[0] = 0

            return -np.sum(np.log(cls._gaussian_lognormal(x, params)))

        # Minimize function
        coeff = op.fmin(
            target_func, param, maxfun=max_fun, maxiter=max_iter, disp=False
        )

        # If clamping mu, also set to 0 in the end, so this value
        # is used for probability calculations
        if clamp_mu:
            coeff[0] = 0

        q = coeff[2]
        # Check if fit worked
        if q >= 1 or np.isinf(q) or np.isneginf(q):
            raise ValueError(
                "No tail, fit failed. q={}".format(q)
            )

        return coeff

    @classmethod
    def _gaussian_lognormal(cls, x, params):
        """
        Gaussian-lognormal mixture probability
        density function.

        Parameters
        ----------
        x : np.array
            Scores for which PDF is calculated
        params : tuple
            Parameters of lognormal-Gaussian mixture
            (mu, sigma, class weight q, loglog, logsigma)

        Returns
        -------
        np.array
            Probabilities
        """
        return cls._gaussian(x, params) + cls._lognormal(x, params)

    @classmethod
    def _gaussian(cls, x, params):
        """
        Normal probability density (multiplied
        by class weight).

        Parameters
        ----------
        x : np.array
            Scores for which PDF is calculated
        params : tuple
            Parameters of lognormal-Gaussian mixture
            (mu, sigma, class weight q, loglog, logsigma)

        Returns
        -------
        np.array
            Probabilities
        """
        mu, sigma, q, logmu, logsigma = params
        return q * stats.norm.pdf(x, loc=mu, scale=sigma)

    @classmethod
    def _lognormal(cls, x, params):
        """
        Log normal probability density (multiplied
        by class weight).

        Parameters
        ----------
        x : np.array
            Scores for which PDF is calculated
        params : tuple
            Parameters of lognormal-Gaussian mixture
            (mu, sigma, class weight q, loglog, logsigma)

        Returns
        -------
        np.array
            Probabilities
        """
        mu, sigma, q, logmu, logsigma = params

        # only assign probability to defined (i.e. positive) values,
        # set all others to zero
        prob = np.zeros(len(x))
        xpos = x > 0
        prob[xpos] = (1 - q) * stats.norm.pdf(
            np.log(x[xpos]), loc=logmu, scale=logsigma
        ) / x[xpos]

        return prob

    def probability(self, x, plot=False):
        """
        Calculate posterior probability of EC pair to
        be located in positive (lognormal) tail of the
        distribution.

        Parameters
        ----------
        x : np.array (or list-like)
            List of scores
        plot : bool, optional (default: False)
            Plot score distribution and probabilities

        Returns
        -------
        posterior : np.array(float)
            Posterior probability of being in signal
            component of mixture model
        """
        x = np.array(x)
        p_lognormal = self._lognormal(x, self.params)
        p_gaussian = self._gaussian(x, self.params)

        posterior = p_lognormal / (p_lognormal + p_gaussian)

        if plot:
            plt.figure(figsize=(12, 8))
            # fig = plt.figure(figsize=(4,4))
            c = "#fdc832"
            n_ECs, edges = np.histogram(x, 1000, density=True)
            mid = []
            for l, r in zip(edges[:-1], edges[1:]):
                mid.append((l + r) / 2)

            plt.plot(
                mid, n_ECs, '-', color=c, markerfacecolor=c,
                markeredgecolor='None', linewidth=1
            )
            plt.plot(x, posterior, '-k', linewidth=2)
            plt.plot(x, p_lognormal, 'r', linewidth=1)
            plt.plot(x, p_gaussian, 'b', linewidth=1)

            takeover = x[p_lognormal > p_gaussian].min()
            plt.axvline(takeover, color="grey")
            plt.axvline(x[posterior > 0.99].min(), color="grey", lw=1)

            pompom = abs(x.min())
            plt.axvline(-pompom, color="grey", ls="--")
            plt.axvline(pompom, color="grey", ls="--")

            plt.xlabel("EC scores")
            plt.ylabel("PDF")

        return posterior


class ScoreMixtureModel:
    """
    Assign to each EC score the probability of being in the
    lognormal tail of a skew normal-lognormal mixture model.
    """
    def __init__(self, x):
        """
        Mixture model of evolutionary coupling scores to
        determine signifcant scores that are in high-scoring,
        positive tail of distribution.

        Parameters
        ----------
        x : np.array (or list-like)
            EC scores from which to infer the mixture model
        """
        x = np.array(x)

        # Infer parameters of mixture model
        self.params = self._learn_params(x)

    @classmethod
    def skewnorm_pdf(cls, x, location, scale, skew):
        """
        Probability density of skew normal distribution
        (noise component)

        Parameters
        ---------
        x : np.array(float)
            Data for which probability density should be calculated
        location : float
            Location parameter of skew normal distribution
        scale : float
            Scale parameter of skew normal distribution
        skew : float
            Skew parameter of skew normal distribution

        Returns
        ------
        density : np.array(float)
            Probability density for input array x
        """
        x_transform = (x - location) / scale
        density = (
            2 / scale *
            stats.norm.pdf(x_transform) *
            stats.norm.cdf(skew * x_transform)
        )

        return density

    @classmethod
    def lognorm_pdf(cls, x, logmu, logsig):
        """
        Probability density of lognormal distribution
        (signal component)

        Parameters
        ---------
        x : np.array(float)
            Data for which probability density should be calculated
        logmu : float
            Location of lognormal distribution (signal component)
        logsig : float
            Scale parameter of lognormal distribution (signal component)

        Returns
        ------
        density : np.array(float)
            Probability density for input array x
        """
        density = np.zeros(len(x))
        xpos = x > 0
        density[xpos] = stats.norm.pdf(
            np.log(x[xpos]), loc=logmu, scale=logsig
        ) / x[xpos]

        return density

    @classmethod
    def skewnorm_constraint(cls, scale, skew):
        """
        Given scale and skew, returns location parameter to yield mean zero

        Parameters
        ----------
        scale : float
            Scale parameter of skew normal distribution
        skew : float
            Skew parameter of skew normal distribution

        Returns
        -------
        location : float
            Location parameter of skew normal distribution s.t.
            mean of distribution is equal to 0
        """
        location = -scale * skew / np.sqrt(1 + skew**2) * np.sqrt(2 / np.pi)
        return location

    @classmethod
    def mixture_pdf(cls, x, p, scale, skew, logmu, logsig):
        """
        Compute mixture probability

        Parameters
        ----------
        x : np.array(float)
            Data for which probability density should be calculated
        p : float
            Mixing fraction between components for noise component
            (signal component will be 1-p)
        scale : float
            Scale parameter of skew normal distribution (noise component)
        skew : float
            Skew parameter of skew normal distribution (noise component)
        logmu : float
            Location of lognormal distribution (signal component)
        logsig : float
            Scale parameter of lognormal distribution (signal component)

        Returns
        -------
        density : np.array(float)
            Probability density for input array x
        """
        location = cls.skewnorm_constraint(scale, skew)
        density = (
            p * cls.skewnorm_pdf(x, location, scale, skew) +
            (1 - p) * cls.lognorm_pdf(x, logmu, logsig)
        )

        return density

    @classmethod
    def posterior_signal(cls, x, p, scale, skew, logmu, logsig):
        """
        Compute posterior probability of being in signal component

        Parameters
        ----------
        x : np.array(float)
            Data for which probability density should be calculated
        p : float
            Mixing fraction between components for noise component
            (signal component will be 1-p)
        scale : float
            Scale parameter of skew normal distribution (noise component)
        skew : float
            Skew parameter of skew normal distribution (noise component)
        logmu : float
            Location of lognormal distribution (signal component)
        logsig : float
            Scale parameter of lognormal distribution (signal component)

        Returns
        -------
        posterior : np.array(float)
            Posterior probability of being in signal component
            for input array x
        """
        P = cls.mixture_pdf(x, p, scale, skew, logmu, logsig)
        posterior = np.zeros(P.shape)
        f2 = cls.lognorm_pdf(x, logmu, logsig)
        posterior[x > 0] = (1 - p) * f2[x > 0] / P[x > 0]

        return posterior

    @classmethod
    def _learn_params(cls, x):
        """
        Infer parameters of mixture model.

        Parameters
        ----------
        x : np.array(float)
            EC scores from which to infer the mixture model

        Returns
        -------
        theta : np.array(float)
            Array with inferred parameters of model
            (mixing fraction, skew-normal scale, skew-normal skew,
            log-normal mean, log-normal stddev)
        """
        # mixing fraction, skew-normal scale, skew-normal skew,
        # log-normal mean, log-normal stddev
        theta = np.array(
            [0.5, np.std(x), 0, np.log(np.max(x)), 0.1]
        )

        def loglk_fun(params):
            return np.sum(np.log(cls.mixture_pdf(x, *params)))

        loglk = loglk_fun(theta)
        delta_loglk = 100
        max_iter = 200
        cur_iter = 0
        tolerance = 0.0001

        while delta_loglk > tolerance and cur_iter < max_iter:
            # E step
            z = 1 - cls.posterior_signal(x, *theta)

            # M step
            # MLE of the mixing fraction is the mean z
            theta[0] = np.mean(z)

            # Log-normal component
            # MLE is the z-weighted mean and std deviation of the log-scores
            pos_ix = x > 0
            z_complement = 1 - z[pos_ix]
            log_score = np.log(x[pos_ix])
            theta[3] = np.sum(z_complement * log_score) / np.sum(z_complement)
            theta[4] = np.sqrt(
                np.sum(z_complement * (log_score - theta[3])**2) / z_complement.sum()
            )

            # Skew-normal distribution
            # MLE requires numerical optimization
            def objfun(params):
                return -np.sum(
                    z * np.log(
                        cls.skewnorm_pdf(
                            x, cls.skewnorm_constraint(params[0], params[1]),
                            params[0], params[1]
                        )
                    )
                )

            theta[1:3] = op.fmin(objfun, theta[1:3], disp=False)

            # Test for EM convergence
            loglk_new = loglk_fun(theta)
            delta_loglk = loglk_new - loglk
            loglk = loglk_new

            # status update
            cur_iter += 1

        return theta

    def probability(self, x, plot=False):
        """
        Calculate posterior probability of EC pair to
        be located in positive (lognormal) tail of the
        distribution.

        Parameters
        ----------
        x : np.array (or list-like)
            List of scores

        Returns
        -------
        posterior : np.array(float)
            Posterior probability of being in signal
            component of mixture model
        """
        posterior = self.posterior_signal(x, *self.params)

        if plot:
            plt.hist(x, normed=True, bins=50, color="k")
            plt.plot(x, self.mixture_pdf(x, *self.params), color="r", lw=3)
            plt.plot(x, posterior, color="gold", lw=3)

        return posterior


def add_mixture_probability(ecs, model="skewnormal", score="cn",
                            clamp_mu=False, plot=False):
    """
    Add lognormal mixture model probability to EC table.

    Parameters
    ----------
    ecs : pd.DataFrame
        EC table with scores
    model : {"skewnormal", "normal"}, optional (default: skewnormal)
        Use model with skew-normal or normal distribution
        for the noise component of mixture model
    score : str, optional (default: "cn")
        Score on which mixture model will be based
    clamp_mu : bool, optional (default: False)
        Fix mean of Gaussian component to 0 instead of
        fitting it based on data
    plot : bool, optional (default: False)
        Plot score distribution and probabilities

    Returns
    -------
    ec_prob : pd.DataFrame
        EC table with additional column "probability"
        that for each EC contains the posterior
        probability of belonging to the lognormal
        tail of the distribution.
    """
    ec_prob = deepcopy(ecs)

    # learn mixture model
    if model == "skewnormal":
        mm = ScoreMixtureModel(ecs.loc[:, score].values)
    elif model == "normal":
        mm = LegacyScoreMixtureModel(ecs.loc[:, score].values, clamp_mu)
    else:
        raise ValueError(
            "Invalid model selection, valid options are: "
            "skewnormal, normal"
        )

    # assign probability
    ec_prob.loc[:, "probability"] = mm.probability(
        ec_prob.loc[:, score].values, plot=plot
    )

    return ec_prob
