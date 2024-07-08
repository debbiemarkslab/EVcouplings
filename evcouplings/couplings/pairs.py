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
  Anna G. Green (EVComplex Score code)
"""

from math import ceil
from copy import deepcopy
from pkg_resources import resource_filename

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as op
from scipy import stats
from sklearn.linear_model import LogisticRegression

from evcouplings.utils.calculations import median_absolute_deviation
from evcouplings.utils.config import read_config_file


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

    stacked_ecs = pd.concat([top_ecs, flipped])

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


class EVComplexScoreModel:
    """
    Assign to each EC score a (unnormalized) EVcomplex score as
    described in Hopf, Schärfe et al. (2014).

    TODO: this implementation currently does not take into account
    score normalization for the number of sequences and length of
    the model
    """
    def __init__(self, x):
        """
        Initialize EVcomplex score model
        
        Parameters
        ----------
        x : np.array (or list-like)
            EC scores from which to infer the mixture model
        """
        self.x = np.array(x)

    def probability(self, x, plot=False):
        """
        Calculates evcomplex score as cn_score / min_cn_score.
        TODO: plotting functionality not yet implemented

        Parameters
        ----------
        x : np.array (or list-like)
            List of scores
        plot: bool, optional (default: False)
            Plot score distribution

        Returns
        -------
        probability: np.array(float)
            EVcomplex score
        """
        # Calculate the minimum score
        min_score = abs(np.min(self.x))

        return x / min_score


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
    elif model == "evcomplex":
        mm = EVComplexScoreModel(ecs.loc[:, score].values)
    else:
        raise ValueError(
            "Invalid model selection, valid options are: "
            "skewnormal, normal, evcomplex"
        )

    # assign probability
    ec_prob.loc[:, "probability"] = mm.probability(
        ec_prob.loc[:, score].values, plot=plot
    )

    return ec_prob


def logreg_classifier_to_dict(classifier, feature_names=None):
    """
    Serialize sklearn logistic regression classifier

    Inspired by https://stackoverflow.com/questions/48328012/python-scikit-learn-to-json

    Parameters
    ----------
    classifier : sklearn.linear_model.LogisticRegression
        Logistic regression classifier to be serialized
    feature_names : list(str)
        Feature names of coefficients in classifier, must
        be in the same order as classifier.coef_

    Returns
    -------
    model : dict
        Serialized classifier
    """
    params = {
        "classifier_settings": classifier.get_params(),
        "model_settings": {}
    }

    for attrib in ["classes_", "intercept_", "coef_", "n_iter_"]:
        params["model_settings"][attrib] = getattr(classifier, attrib).tolist()

    if feature_names is not None:
        params["feature_names"] = feature_names

    return params


def logreg_classifier_from_dict(params):
    """
    Deserialize model parameters into sklearn LogisticRegression classifier

    Inspired by https://stackoverflow.com/questions/48328012/python-scikit-learn-to-json

    Parameters
    -------
    model : dict
        Serialized classifier

    Parameters
    ----------
    classifier : sklearn.linear_model.LogisticRegression
        Deserialized logistic regression classifier
    feature_names : list(str)
        Feature names of coefficients in classifier
        (in same order as classifier.coef_)
    """
    classifier = LogisticRegression(
        **params["classifier_settings"]
    )

    for attrib, values in params["model_settings"].items():
        setattr(classifier, attrib, np.array(values))

    feature_names = params.get("feature_names")

    return classifier, feature_names


def add_freqs_to_ec_table(ecs, freqs):
    """
    Add residue and gap frequency as well as conservation info
    to data table by merging EC table and frequencies table

    Parameters
    ----------
    ecs : pd.DataFrame
        EC table generated by pipeline (e.g. _ECs.txt, _CouplingsScores.csv)
    freqs : pd.DataFrame
        Frequency table generated by pipeline (_frequencies.csv) using
        evcouplings.align.protocol.describe_frequencies()
    """
    # rename column names to be more descriptive;
    # also get rid of rows with nan values (these are present only if input
    # alignment has lowercase letters but will cause problems downstream
    # where we need frequencies for uppercase letters only. Hence, remove
    # from table right away of freq_i assignment below will crash.
    freqs = freqs.rename(
        columns={
            "-": "gap_i",
            "conservation": "cons_i",
        }
    ).dropna()

    # extract frequency of target residue for each position
    freqs.loc[:, "freq_i"] = [r[r["A_i"]] for idx, r in freqs.iterrows()]

    # limit to relevant columns and create second version
    # for mapping second position suffixed with _j
    freqs_sel_i = freqs[["i", "A_i", "freq_i", "gap_i", "cons_i"]]

    freqs_sel_j = freqs_sel_i.rename(columns={
        c: c.replace("i", "j") for c in freqs_sel_i.columns
    })

    # add frequency/conservation information to table
    ecs_with_freqs = ecs.merge(
        freqs_sel_i, on=["i", "A_i"]
    ).merge(
        freqs_sel_j, on=["j", "A_j"]
    )

    # check nothing went wrong and we kept full table
    assert len(ecs_with_freqs) == len(ecs)

    return ecs_with_freqs


def mad_outlier_score(x):
    """
    Median absolute deviation outlier scoring of
    ECs in tail vs. background noise distribution
    (robust z-scoring using median and median
    absolute deviation instead of mean and
    sample standard deviation)

    Parameters
    ----------
    x : np.array
        List of scores (note this must contain *all*
        scores for estimation of median and median
        absolute deviation, i.e. a list of top X
        ECs only will give wrong results)

    Returns
    -------
    np.array
        Outlier scores
    """
    med = np.median(x)
    mad = median_absolute_deviation(x)
    return (x - med) / mad


class LogisticRegressionScorer:
    """
    Rescore ECs based on logistic regression model
    fitted to large set of runs
    """
    def __init__(self, logreg_model_file=None, min_n_eff_over_l=0.375):
        """
        Create new logistic regression-based
        EC rescorer

        Parameters
        ----------
        logreg_model_file : str, optional (default: None)
            Specify path to yml file with logistic regression
            model parameters; if None, will use default
            model included with package
            (evcouplings/couplings/scoring_models/logistic_regression_all.yml)
        min_n_eff_over_l : float, optional (default: 0.3)
            Minimum number of effective sequences per model site required
            for rescoring to be applied; otherwise standard score will
            be returned and all probabilities will be set to 0. The
            default value will be divided by theta for the rescored run,
            the default of 0.375 derives from N_eff/L = 0.3 at theta = 0.8
        """
        # by default load internal classifier included with package
        if logreg_model_file is None:
            logreg_model_file = resource_filename(
                __name__, "scoring_models/logistic_regression_all.yml"
            )

        # load classifier from param file
        logreg_model_serialized = read_config_file(logreg_model_file)

        # deserialize and store classifier
        self.classifier, self.feature_names = logreg_classifier_from_dict(
            logreg_model_serialized
        )

        # store min N_eff/L requirement
        self.min_n_eff_over_l = min_n_eff_over_l

    @classmethod
    def _create_full_data_table(cls, ecs, freqs, theta, effective_sequences, num_sites):
        """
        Create full EC feature table

        Parameters
        ----------
        ecs : pd.DataFrame
            Full, unfiltered EC table
        freqs : pd.DataFrame
            Frequencies table
        theta : float
            Theta parameter used for sequence reweighting
            (used to adjust influence of different thetas in scoring model)
        effective_sequences : float
            Number of effective sequences after clustering and downweighting
        num_sites : float, optional (default: None)
            Number of sites in model / alignment; if None, will
            be automatically inferred from Ec table

        Returns
        -------
        pd.DataFrame
            Full EC table with annotated input features for model
        """
        meff_over_l = effective_sequences / num_sites
        meff_over_l2 = effective_sequences / num_sites ** 2

        meff_over_l_norm = meff_over_l / theta
        meff_over_l2_norm = meff_over_l2 / theta

        # add frequency and conservation info to EC table
        ecs = add_freqs_to_ec_table(
            ecs, freqs
        )

        # assign additional features to EC table
        # needed by prediction model
        ecs = ecs.assign(
            num_sites_log=np.log10(num_sites),
            min_gap=np.minimum(ecs.gap_i, ecs.gap_j),
            max_gap=np.maximum(ecs.gap_i, ecs.gap_j),
            min_cons=np.minimum(ecs.cons_i, ecs.cons_j),
            max_cons=np.maximum(ecs.cons_i, ecs.cons_j),
            meff_over_l_norm_log=np.log10(meff_over_l_norm),
            meff_over_l2_norm_log=np.log10(meff_over_l2_norm)
        )

        return ecs

    def score(self, ecs, freqs, theta, effective_sequences, num_sites=None, score="cn"):
        """
        Rescore EC table (must be full, unfiltered table) using logistic regression
        based probabilistic scoring. If number of effective sequences is too low for reliable
        scoring, score will default to input score and probabilities will all be set to 0.

        Parameters
        ----------
        ecs : pd.DataFrame
            Full, unfiltered EC table
        freqs : pd.DataFrame
            Frequencies table
        theta : float
            Theta parameter used for sequence reweighting
            (used to adjust influence of different thetas in scoring model)
        effective_sequences : float
            Number of effective sequences after clustering and downweighting
        num_sites : float, optional (default: None)
            Number of sites in model / alignment; if None, will
            be automatically inferred from Ec table
        score : string, optional (default: "cn")
            Input score to be input into logistic regression model.
            Model has been trained on CN scores only, so is not applicable
            to other types of scoring like DI score

        Returns
        -------
        ecs_final : pd.DataFrame
            Rescored ECs, same table as ecs but with three
            additional columns (mad_score, probability, score) containing
            1) median absolute deviation outlier score, 2) logistic regression
            model probability for EC to be true EC, and 3) pair score
            derived from decision function of logistic regression model
            (this can reorder the original ECs based on positional conservation
            and gap frequencies).
            Final table is sorted by "score" in descending order.
        """
        # infer num_sites from EC table if not defined
        if num_sites is None:
            num_sites = len(
                set(ecs.i.unique()) | set(ecs.j.unique())
            )

        # if N_eff/L is too low for model to give reliable results,
        # keep standard score and set all probabilities to 0
        if effective_sequences / num_sites / theta < self.min_n_eff_over_l:
            return ecs.assign(
                score=ecs[score],
                probability=0
            )

        # add medium absolute deviation outlier score to copy of EC table
        ecs = ecs.assign(
            mad_score=mad_outlier_score(ecs[score])
        )

        # add additional information to table
        ecs_full = self._create_full_data_table(
            ecs, freqs, theta, effective_sequences, num_sites
        )

        # extract features for classifier ("X")
        feature_table = ecs_full.reindex(self.feature_names, axis=1)

        # predict probabilities of contact and decision function
        # (target class/true EC := 1);
        # note: apply to .values to avoid sklearn warnings that model does not have feature names
        probs = self.classifier.predict_proba(feature_table.values)[:, 1]
        decision_func = self.classifier.decision_function(feature_table.values)

        # assign to EC table
        ecs_final = ecs_full.assign(
            score=decision_func,
            probability=probs
        ).sort_values(
            by="score", ascending=False
        )

        # remove temporary feature columns and return updated EC table
        return ecs_final[
            list(ecs.columns) + ["probability", "score"]
        ]
