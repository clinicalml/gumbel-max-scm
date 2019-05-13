import numpy as np, random
from .MDP import MDP
from .State import State
from .Action import Action
from tqdm import tqdm_notebook as tqdm

'''
Simulates data generation from an MDP
'''
class DataGenerator(object):

    def select_actions(self, state, policy):
        '''
        select action for state from policy
        if unspecified, a random action is returned
        '''
        if state not in policy:
            return Action(action_idx = np.random.randint(8))
        return policy[state]

    def simulate(self, num_iters, max_num_steps,
            policy=None, policy_idx_type='full', p_diabetes=0.2,
            output_state_idx_type='obs', use_tqdm=False, tqdm_desc=''):
        '''
        policy is an array of probabilities
        '''
        assert policy is not None, "Please specify a policy"

        # Set the default value of states / actions to negative -1,
        # corresponding to None
        iter_states = np.ones((num_iters, max_num_steps+1, 1), dtype=int)*(-1)
        iter_actions = np.ones((num_iters, max_num_steps, 1), dtype=int)*(-1)
        iter_rewards = np.zeros((num_iters, max_num_steps, 1))
        iter_lengths = np.zeros((num_iters, 1), dtype=int)

        # Record diabetes, the hidden mixture component
        iter_component = np.zeros((num_iters, max_num_steps, 1), dtype=int)
        mdp = MDP(init_state_idx=None, # Random initial state
                  policy_array=policy, policy_idx_type=policy_idx_type,
                  p_diabetes=p_diabetes)

        # Empirical transition / reward matrix
        if output_state_idx_type == 'obs':
            emp_tx_mat = np.zeros((Action.NUM_ACTIONS_TOTAL,
                State.NUM_OBS_STATES, State.NUM_OBS_STATES))
            emp_r_mat = np.zeros((Action.NUM_ACTIONS_TOTAL,
                State.NUM_OBS_STATES, State.NUM_OBS_STATES))
        elif output_state_idx_type == 'full':
            emp_tx_mat = np.zeros((Action.NUM_ACTIONS_TOTAL,
                State.NUM_FULL_STATES, State.NUM_FULL_STATES))
            emp_r_mat = np.zeros((Action.NUM_ACTIONS_TOTAL,
                State.NUM_FULL_STATES, State.NUM_FULL_STATES))
        else:
            raise NotImplementedError()

        for itr in tqdm(range(num_iters), disable=not(use_tqdm), desc=tqdm_desc):
            # MDP will generate the diabetes index as well
            mdp.state = mdp.get_new_state()
            this_diabetic_idx = mdp.state.diabetic_idx
            iter_component[itr, :] = this_diabetic_idx  # Never changes
            iter_states[itr, 0, 0] = mdp.state.get_state_idx(
                    idx_type=output_state_idx_type)
            for step in range(max_num_steps):
                step_action = mdp.select_actions()

                this_action_idx = step_action.get_action_idx().astype(int)
                this_from_state_idx = mdp.state.get_state_idx(
                        idx_type=output_state_idx_type).astype(int)

                # Take the action, new state is property of the MDP
                step_reward = mdp.transition(step_action)
                this_to_state_idx = mdp.state.get_state_idx(
                        idx_type=output_state_idx_type).astype(int)

                iter_actions[itr, step, 0] = this_action_idx
                iter_states[itr, step+1, 0] = this_to_state_idx

                # Record in transition matrix
                emp_tx_mat[this_action_idx,
                       this_from_state_idx, this_to_state_idx] += 1
                emp_r_mat[this_action_idx,
                       this_from_state_idx, this_to_state_idx] += step_reward

                if step_reward != 0:
                    iter_rewards[itr, step, 0] = step_reward
                    iter_lengths[itr, 0] = step+1
                    break

            if step == max_num_steps-1:
                iter_lengths[itr, 0] = max_num_steps

        return iter_states, iter_actions, iter_lengths, iter_rewards, iter_component, emp_tx_mat, emp_r_mat
