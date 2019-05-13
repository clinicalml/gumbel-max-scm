"""
counterfactual
"""
import numpy as np
import mdptoolboxSrc.mdp as mdptools
import warnings
import cf.gumbelTools as gt
from tqdm import tqdm_notebook as tqdm

class MatrixMDP(object):
    def __init__(self, tx_mat, r_mat, p_initial_state=None, p_mixture=None):
        """__init__

        :param tx_mat:  Transition matrix of shape (n_components x n_actions x
        n_states x n_states) or (n_actions x n_states x n_states)
        :param r_mat:  Reward matrix of shape (n_components x n_actions x
        n_states x n_states) or (n_actions x n_states x n_states)
        :param p_initial_state: Probability over initial states
        :param p_mixture: Probability over "mixture" components, in this case
        diabetes status
        """
        # QA the size of the inputs
        assert tx_mat.ndim == 4 or tx_mat.ndim == 3, \
            "Transition matrix wrong dims ({} != 3 or 4)".format(tx_mat.ndim)
        assert r_mat.ndim == 4 or r_mat.ndim == 3, \
            "Reward matrix wrong dims ({} != 3 or 4)".format(tx_mat.ndim)
        assert r_mat.shape == tx_mat.shape, \
            "Transition / Reward matricies not the same shape!"
        assert tx_mat.shape[-1] == tx_mat.shape[-2], \
            "Last two dims of Tx matrix should be equal to num of states"

        # Get the number of actions and states
        n_actions = tx_mat.shape[-3]
        n_states = tx_mat.shape[-2]

        # Get the number of components in the mixture:
        # If no hidden component, add a dummy so the rest of the interface works
        if tx_mat.ndim == 3:
            n_components = 1
            tx_mat = tx_mat[np.newaxis, ...]
            r_mat = r_mat[np.newaxis, ...]
        else:
            n_components = tx_mat.shape[0]

        # Get the prior over initial states
        if p_initial_state is not None:
            if p_initial_state.ndim == 1:
                p_initial_state = p_initial_state[np.newaxis, :]

            assert p_initial_state.shape == (n_components, n_states), \
                ("Prior over initial state is wrong shape "
                 "{} != (C x S)").format(p_initial_state.shape)

        # Get the prior over components
        if n_components == 1:
            p_mixture = np.array([1.0])
        elif p_mixture is not None:
            assert p_mixture.shape == (n_components, ), \
                ("Prior over components is wrong shape "
                 "{} != (C)").format(p_mixture.shape)

        self.n_components = n_components
        self.n_actions = n_actions
        self.n_states = n_states
        self.tx_mat = tx_mat
        self.r_mat = r_mat
        self.p_initial_state = p_initial_state
        self.p_mixture = p_mixture

        self.current_state = None
        self.component = None

    def reset(self):
        """reset

        Reset the environment, and return the initial position

        :returns: Tuple of (initial state, component)
        """
        # Draw from the mixture
        if self.p_mixture is None:
            self.component = np.random.randint(self.n_components)
        else:
            self.component = np.random.choice(
                self.n_components, size=1, p=self.p_mixture.tolist())[0]

        # Draw an initial state
        if self.p_initial_state is None:
            self.current_state = np.random.randint(self.n_states)
        else:
            self.current_state = np.random.choice(
                self.n_states, size=1,
                p=self.p_initial_state[self.component, :].squeeze().tolist())[0]

        return self.current_state, self.component

    def step(self, action):
        """step

        Take a step with the given action

        :action: Integer of the action
        :returns: Tuple of (next_state, reward)
        """
        assert action in range(self.n_actions), "Invalid action!"
        is_term = False

        next_prob = self.tx_mat[
                self.component, action, self.current_state,
                :].squeeze()

        assert np.isclose(next_prob.sum(), 1), "Probs do not sum to 1!"

        next_state = np.random.choice(self.n_states, size=1, p=next_prob)[0]

        reward = self.r_mat[self.component, action,
                            self.current_state, next_state]
        self.current_state = next_state

        # In this MDP, rewards are only received at the terminal state
        if reward != 0:
            is_term = True

        return self.current_state, reward, is_term

    def policyIteration(self, discount=0.9, obs_pol=None, skip_check=False,
            eval_type=1):
        """Calculate the optimal policy for the marginal tx_mat and r_mat,
        using policy iteration from pymdptoolbox

        Note that this function marginalizes over any mixture components if
        they exist.

        :discount: Discount factor for rewards
        :returns: Policy matrix with deterministic policy

        """
        # Define the marginalized transition and reward matrix
        r_mat_obs = self.r_mat.T.dot(self.p_mixture).T
        tx_mat_obs = self.tx_mat.T.dot(self.p_mixture).T

        # Run Policy Iteration
        pi = mdptools.PolicyIteration(
            tx_mat_obs, r_mat_obs, discount=discount, skip_check=skip_check,
            policy0=obs_pol, eval_type=eval_type)
        pi.setSilent()
        pi.run()

        # Convert this (deterministic) policy pi into a matrix format 
        pol_opt = np.zeros((self.n_states, self.n_actions))
        pol_opt[np.arange(len(pi.policy)), pi.policy] = 1

        return pol_opt

class BatchSampler(object):
    """BatchSampler

    Samples batches of episodes
    """
    def __init__(self, mdp):
        assert isinstance(mdp, MatrixMDP), "mdp argument must be a MatrixMDP"
        self.mdp = mdp

    def on_policy_sample(self, policy=None, n_steps=10, n_samps=1, out='array', 
            use_tqdm=False, tqdm_desc=''):
        """on_policy_sample.

        :param policy: Stochastic matrix of size (n_states x n_actions), default is random policy
        :param n_steps: Maximum length of an episode
        :param n_samps: Number of episodes in the batch
        :param out: (Not implemented) type of output, must be 'array' for now
        :param use_tqdm: Whether or not to display progress bars
        :param tqdm_desc: Description for progress bars
        :returns: Array containing samples collected under the policy
        """
        if policy is not None:
            assert policy.shape == (self.mdp.n_states, self.mdp.n_actions), \
                "Policy is the wrong shape.  {} != S x A".format(policy.shape)

        # For each trajectory, for each step, we record
        # t, A_{t}, O_{t}, O_{t+1}, h_{t}, h_{t+1}, R_{t}
        # Note that in the toy example, "h" corresponds to the hidden component

        assert out == 'array', "Only 'array' supported as output type for now"
        result = np.zeros((n_samps, n_steps, 7))
        result[:, :, 1:4] = -1  # Placeholder for tracking the end of the seq

        for samp_idx in tqdm(range(n_samps),
                             disable=not(use_tqdm), desc=tqdm_desc):
            current_state, component = self.mdp.reset()

            # Sample the trajectory
            for time_idx in range(n_steps):
                if policy is None:  # Random Policy
                    this_action = np.random.randint(self.mdp.n_actions)
                else:
                    this_action = np.random.choice(
                        self.mdp.n_actions, size=1,
                        p=policy[current_state, :].squeeze().tolist())[0]

                # Terminal state if the reward is nonzero
                next_state, this_reward, is_term = self.mdp.step(this_action)

                # Record State
                result[samp_idx, time_idx] = (
                    time_idx,
                    this_action,
                    current_state,
                    next_state,
                    component,
                    component,
                    this_reward)

                current_state = next_state
                if is_term:
                    break

        return result

    def cf_trajectory(self, batch, cf_policy, n_cf_samps=1,
            use_tqdm=False, tqdm_desc=''):
        """cf_trajectory

        :param batch: Output of the sampler, shape is (n_samps, n_steps, 7)
        :param cf_policy: Counterfactual policy to evaluate
        :param n_cf_samps: Counterfactual samples to draw per episode
        :param use_tqdm: Whether or not to display progress bars
        :param tqdm_desc: Description for progress bars

        :returns: Array containing counterfactual trajectories
        """

        # Used for Monte Carlo sampling
        n_draws = 1000

        # For each trajectory, for each step, we record
        # t, A_{t}, O_{t}, O_{t+1}, h_{t}, h_{t+1}, R_{t}
        # Note that in the toy example, "h" corresponds to the hidden component
        n_obs_eps = batch.shape[0]
        n_obs_steps = batch.shape[1]

        # Result matrix has an extra dimension for number of CF draws per OBS
        result = np.zeros((n_obs_eps, n_cf_samps, n_obs_steps, 7))
        result[:, :, :, 0] = np.arange(n_obs_steps)
        result[:, :, :, 1:4] = -1  # Placeholders for end of sequence

        # Take posterior over the mixture components in batch form
        # NOTE: This code does not serve a purpose in our current toy example,
        # because we define the MDP with a single component, but it could be
        # used in a future experiment with a single time-independent confounder
        if self.mdp.n_components == 1:
            mx_posterior = np.ones((n_obs_eps, 1))
        else:
            mx_posterior = self.mixture_posterior(batch)

        for obs_samp_idx in tqdm(range(n_obs_eps), disable=not(use_tqdm), desc=tqdm_desc):
            for cf_samp_idx in range(n_cf_samps):
                obs_actions = batch[obs_samp_idx, :, 1].astype(int).squeeze().tolist()
                obs_from_states = batch[obs_samp_idx, :, 2].astype(int).squeeze().tolist()
                obs_to_states = batch[obs_samp_idx, :, 3].astype(int).squeeze().tolist()

                # Same initial state
                current_state = obs_from_states[0]

                # Infer / Sample from the mixture posterior
                this_mx_posterior = mx_posterior[obs_samp_idx].tolist()
                component = np.random.choice(
                    self.mdp.n_components, size=1, p=this_mx_posterior)

                for time_idx in range(n_obs_steps):
                    obs_action = obs_actions[time_idx]

                    if cf_policy is None:  # Random Policy
                        cf_action = np.random.randint(self.mdp.n_actions)
                    else:
                        cf_action = np.random.choice(
                            self.mdp.n_actions, size=1,
                            p=cf_policy[current_state, :].squeeze().tolist())[0]

                    # Interventional probabilities under new action
                    new_interv_probs = \
                        self.mdp.tx_mat[component,
                                        cf_action, current_state,
                                        :].squeeze().tolist()

                    # If observed sequence did not terminate, then infer cf
                    # probabilities;  Otherwise treat this as an interventional
                    # query (once we're past the final time-step of the
                    # observed sequence, there is no posterior over latents)

                    if obs_action == -1:
                        cf_probs = new_interv_probs
                    else:
                        # Old and new interventional probabilities
                        prev_interv_probs = \
                            self.mdp.tx_mat[component,
                                            obs_action, obs_from_states[time_idx],
                                            :].squeeze().tolist()

                        assert prev_interv_probs[obs_to_states[time_idx]] != 0

                        # Infer counterfactual probabilities
                        cf_probs = tx_posterior(
                            prev_interv_probs, new_interv_probs,
                            obs=obs_to_states[time_idx],
                            n_samp=n_draws).tolist()

                    next_state = np.random.choice(
                        self.mdp.n_states, size=1, p=cf_probs)[0]
                    this_reward = self.mdp.r_mat[
                        component, cf_action, current_state, next_state]

                    # Record result
                    result[obs_samp_idx, cf_samp_idx, time_idx] = (
                        time_idx,
                        cf_action,
                        current_state,
                        next_state,
                        component,
                        component,
                        this_reward)

                    if this_reward != 0 and time_idx != n_obs_steps - 1:
                        # Fill in next state, convention in obs_samps
                        result[obs_samp_idx, cf_samp_idx, time_idx + 1] = (
                            time_idx + 1,
                            -1,
                            next_state,
                            -1,
                            component,
                            component,
                            0)
                        break

                    current_state = next_state

        return result

    def mixture_posterior(self, batch):
        """mixture_posterior
        Infer the posterior over the mixture components of the MDP

        :param batch: Batch of observed trajectories (n_samps x n_steps x 7)

        :returns: Posterior over mixture components (n_samps x n_components)
        """
        n_samps = batch.shape[0]
        n_steps = batch.shape[1]
        posterior = np.zeros((n_samps, self.mdp.n_components))

        # Ignore errors due to zeros
        with np.errstate(divide='ignore'):
            log_p_initial_state = np.log(self.mdp.p_initial_state)
            log_p_mixture = np.log(self.mdp.p_mixture)
            log_mat = np.log(self.mdp.tx_mat)

        for obs_samp_idx in range(n_samps):

            # Prior
            this_log_posterior = log_p_mixture.copy()

            # Recall that batch is of size (n_samps x n_steps x 7) with cols:
            # t, A_{t}, O_{t}, O_{t+1}, h_{t}, h_{t+1}, R_{t}

            # Update with likelihood of initial state
            this_log_posterior += log_p_initial_state[
                :, batch[obs_samp_idx, 0, 2].astype(int)]

            for time_idx in range(n_steps):
                # Stop when we reach the end of the sequence
                if batch[obs_samp_idx, time_idx, 1] == -1:
                    break
                # Update likelihood for observed transitions
                this_log_posterior += log_mat[
                    :,  # Across components
                    batch[obs_samp_idx, time_idx, 1].astype(int),  # Action taken
                    batch[obs_samp_idx, time_idx, 2].astype(int),  # From this state
                    batch[obs_samp_idx, time_idx, 3].astype(int)   # To this state
                    ]

            # Convert to normalized probabilities
            this_posterior = np.exp(this_log_posterior)
            try:
                this_posterior = this_posterior / this_posterior.sum(axis=0)
            except RuntimeWarning:
                import pdb
                pdb.set_trace()

            posterior[obs_samp_idx] = this_posterior

        return posterior

def tx_posterior(p_c, p_t, obs=0, n_samp=1000):
    """tx_posterior

    Get a posterior over counterfactual transitions

    :param p_c: "Control" probabilities, under observed action
    :param p_t: "Treatment" probabilities, under different action
    :param obs: Observed outcome under observed action
    :param n_samp: Number of Monte Carlo samples from posterior
    """
    assert isinstance(p_c, list), "Pass probabilities in as a list!"
    assert isinstance(p_t, list), "Pass probabilities in as a list!"

    n_cat = len(p_c)
    assert len(p_c) == len(p_t)
    assert obs in range(n_cat), "Obs is {}, not valid!".format(obs)
    np.testing.assert_approx_equal(np.sum(p_c), 1)
    np.testing.assert_approx_equal(np.sum(p_t), 1)

    # Define our categorical logits
    with np.errstate(divide='ignore'):
        logits_control = np.log(np.array(p_c))
        logits_treat = np.log(np.array(p_t))

    assert p_c[obs] != 0, "Probability of observed event was zero!"

    # Note:  These are the Gumbel values (just g), not log p + g
    posterior_samp = gt.topdown(logits_control, obs, n_samp)

    # The posterior under control should give us the same result as the obs
    assert ((posterior_samp + logits_control).argmax(axis=1) == obs).sum() == n_samp

    # Counterfactual distribution
    # This throws a RunTimeWarning because logits_treat includes some -inf, but
    # that is expected
    posterior_sum = posterior_samp + logits_treat

    # Because some logits are -inf, some entries of posterior_sum will be nan,
    # but this is OK - these correspond to zero-probability transitions.  We
    # just assert here that at least one of the entries for each sample is an 
    # actual number (read the assert below as:  Make sure that none of the 
    # samples have all NaNs)
    assert not np.any(np.all(np.isnan(posterior_sum), axis=1))
    posterior_treat = posterior_sum.argmax(axis=1)

    # Reshape posterior argmax into a 1-D one-hot encoding for each sample
    mask = np.zeros(posterior_sum.shape)
    mask[np.arange(len(posterior_sum)), posterior_treat] = 1
    posterior_prob = mask.sum(axis=0) / mask.shape[0]

    return posterior_prob

def calc_reward(obs_samps, discount=0.9):
    # Column 0 is a time index, column 6 is the reward
    discounted_reward = (discount**obs_samps[..., 0] * obs_samps[..., 6])
    return discounted_reward.sum(axis=-1)  # Take the last axis

def eval_on_policy(obs_samps, discount=0.9, bootstrap=False, n_bootstrap=None):
    """eval_on_policy

    :param obs_samps:
    :param discount:
    :param bootstrap:
    :param n_bootstrap:
    """
    obs_rewards = calc_reward(obs_samps, discount).squeeze()  # 1D array
    assert obs_rewards.ndim == 1

    if bootstrap:
        assert n_bootstrap is not None, "Please specify n_bootstrap"
        bs_rewards = np.random.choice(
            obs_rewards,
            size=(n_bootstrap, obs_rewards.shape[0]),
            replace=True)
        return bs_rewards.mean(axis=1)
    else:
        return obs_rewards.mean()

def eval_wis(obs_samps, obs_policy, new_policy,
                    discount=0.9, bootstrap=False, n_bootstrap=None):
    """eval_off_policy

    Weighted Importance Sampling for Off Policy Evaluation

    :obs_samps: Observed samples
    :policy: Stochastic policy to evaluate
    :returns: Expected returns (scalar)
    """
    # Check dimensions
    assert obs_policy.ndim == 2
    assert new_policy.ndim == 2
    assert obs_samps.ndim == 3

    # Precompute the discounted rewards and importance weights
    obs_rewards = calc_reward(obs_samps, discount).squeeze()  # 1D array
    assert obs_rewards.ndim == 1

    obs_actions = obs_samps[..., 1].astype(int)
    obs_states = obs_samps[..., 2].astype(int)

    # NOTE: This fails silently in that action = -1 corresponds to the end
    # of a sequence, but in this indexing will just take the last action.
    # This is corrected in the next code block "deal with variable length..."
    p_obs = obs_policy[obs_states, obs_actions]
    p_new = new_policy[obs_states, obs_actions]

    # Deal with variable length sequences by setting ratio to 1
    terminated_idx = obs_actions == -1
    p_obs[terminated_idx] = 1
    p_new[terminated_idx] = 1

    if not np.all(p_obs > 0):
        import pdb
        pdb.set_trace()
    assert np.all(p_obs > 0), "Some actions had zero prob under p_obs, WIS fails"

    cum_ir = (p_new / p_obs).prod(axis=1)

    wis_idx = (cum_ir > 0)

    if wis_idx.sum() == 0:
        print("Found zero matching WIS samples, continuing")
        return np.nan, wis_idx, wis_idx.sum()

    if bootstrap:
        assert n_bootstrap is not None, "Please specify n_bootstrap"
        # Get indices, because we need to sample from cum_ir and rewards
        idx = np.random.choice(
                np.arange(obs_rewards.shape[0]),
                size=(n_bootstrap, obs_rewards.shape[0]),
                replace=True)

        # Keepdims so that we can broadcast
        wis_bs_samps = (cum_ir[idx] /
                cum_ir[idx].mean(axis=1, keepdims=True)) * obs_rewards[idx]

        if np.any(np.isnan(wis_bs_samps)):
            import pdb
            pdb.set_trace()

        # Return WIS, one per row
        wis_est = wis_bs_samps.mean(axis=1)
        return wis_est, wis_idx[idx], wis_idx[idx].sum()
    else:
        wis = (cum_ir / cum_ir.mean()) * obs_rewards
        if np.any(np.isnan(wis)):
            import pdb
            pdb.set_trace()

        wis_est = wis.mean()
        return wis_est, wis_idx, wis_idx.sum()
