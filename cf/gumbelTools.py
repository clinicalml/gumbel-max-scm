'''
Tools for sampling efficiently from a Gumbel posterior

Original code taken from https://cmaddis.github.io/gumbel-machinery, and then
modified to work as numpy arrays, and to fit our nomenclature, e.g.
* np.log(alpha) is replaced by log probabilities (which we refer to as logits)
* np.log(sum(alphas)) is removed, because it should always equal zero
'''
import numpy as np

def truncated_gumbel(logit, truncation):
    """truncated_gumbel

    :param logit: Location of the Gumbel variable (e.g., log probability)
    :param truncation: Value of Maximum Gumbel
    """
    # Note: In our code, -inf shows up for zero-probability events, which is
    # handled in the topdown function
    assert not np.isneginf(logit)

    gumbel = np.random.gumbel(size=(truncation.shape[0])) + logit
    trunc_g = -np.log(np.exp(-gumbel) + np.exp(-truncation))
    return trunc_g

def topdown(logits, k, nsamp=1):
    """topdown

    Top-down sampling from the Gumbel posterior

    :param logits: log probabilities of each outcome
    :param k: Index of observed maximum
    :param nsamp: Number of samples from gumbel posterior
    """
    np.testing.assert_approx_equal(np.sum(np.exp(logits)), 1), "Probabilities do not sum to 1"
    ncat = logits.shape[0]

    gumbels = np.zeros((nsamp, ncat))

    # Sample top gumbels
    topgumbel = np.random.gumbel(size=(nsamp))

    for i in range(ncat):
        # This is the observed outcome
        if i == k:
            gumbels[:, k] = topgumbel - logits[i]
        # These were the other feasible options (p > 0)
        elif not(np.isneginf(logits[i])):
            gumbels[:, i] = truncated_gumbel(logits[i], topgumbel) - logits[i]
        # These have zero probability to start with, so are unconstrained
        else:
            gumbels[:, i] = np.random.gumbel(size=nsamp)

    return gumbels
