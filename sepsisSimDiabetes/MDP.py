import numpy as np
from .State import State
from .Action import Action

'''
Includes blood glucose level proxy for diabetes: 0-3
    (lo2, lo1, normal, hi1, hi2); Any other than normal is "abnormal"
Initial distribution:
    [.05, .15, .6, .15, .05] for non-diabetics and [.01, .05, .15, .6, .19] for diabetics

Effect of vasopressors on if diabetic:
    raise blood pressure: normal -> hi w.p. .9, lo -> normal w.p. .5, lo -> hi w.p. .4
    raise blood glucose by 1 w.p. .5

Effect of vasopressors off if diabetic:
    blood pressure falls by 1 w.p. .05 instead of .1
    glucose does not fall - apply fluctuations below instead

Fluctuation in blood glucose levels (IV/insulin therapy are not possible actions):
    fluctuate w.p. .3 if diabetic
    fluctuate w.p. .1 if non-diabetic
Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4530321/

Additional fluctuation regardless of other changes
This order is applied:
    antibiotics, ventilation, vasopressors, fluctuations
'''

class MDP(object):

    def __init__(self, init_state_idx=None, init_state_idx_type='obs',
            policy_array=None, policy_idx_type='obs', p_diabetes=0.2):
        '''
        initialize the simulator
        '''
        assert p_diabetes >= 0 and p_diabetes <= 1, \
                "Invalid p_diabetes: {}".format(p_diabetes)
        assert policy_idx_type in ['obs', 'full', 'proj_obs']

        # Check the policy dimensions (states x actions)
        if policy_array is not None:
            assert policy_array.shape[1] == Action.NUM_ACTIONS_TOTAL
            if policy_idx_type == 'obs':
                assert policy_array.shape[0] == State.NUM_OBS_STATES
            elif policy_idx_type == 'full':
                assert policy_array.shape[0] == \
                        State.NUM_HID_STATES * State.NUM_OBS_STATES
            elif policy_idx_type == 'proj_obs':
                assert policy_array.shape[0] == State.NUM_PROJ_OBS_STATES

        # p_diabetes is used to generate random state if init_state is None
        self.p_diabetes = p_diabetes
        self.state = None

        # Only need to use init_state_idx_type if you are providing a state_idx!
        self.state = self.get_new_state(init_state_idx, init_state_idx_type)

        self.policy_array = policy_array
        self.policy_idx_type = policy_idx_type  # Used for mapping the policy to actions

    def get_new_state(self, state_idx = None, idx_type = 'obs', diabetic_idx = None):
        '''
        use to start MDP over.  A few options:

        Full specification:
        1. Provide state_idx with idx_type = 'obs' + diabetic_idx
        2. Provide state_idx with idx_type = 'full', diabetic_idx is ignored
        3. Provide state_idx with idx_type = 'proj_obs' + diabetic_idx*

        * This option will set glucose to a normal level

        Random specification
        4. State_idx, no diabetic_idx: Latter will be generated
        5. No state_idx, no diabetic_idx:  Completely random
        6. No state_idx, diabetic_idx given:  Random conditional on diabetes
        '''
        assert idx_type in ['obs', 'full', 'proj_obs']
        option = None
        if state_idx is not None:
            if idx_type == 'obs' and diabetic_idx is not None:
                option = 'spec_obs'
            elif idx_type == 'obs' and diabetic_idx is None:
                option = 'spec_obs_no_diab'
                diabetic_idx = np.random.binomial(1, self.p_diabetes)
            elif idx_type == 'full':
                option = 'spec_full'
            elif idx_type == 'proj_obs' and diabetic_idx is not None:
                option = 'spec_proj_obs'
        elif state_idx is None and diabetic_idx is None:
            option = 'random'
        elif state_idx is None and diabetic_idx is not None:
            option = 'random_cond_diab'

        assert option is not None, "Invalid specification of new state"

        if option in ['random', 'random_cond_diab'] :
            init_state = self.generate_random_state(diabetic_idx)
            # Do not start in death or discharge state
            while init_state.check_absorbing_state():
                init_state = self.generate_random_state(diabetic_idx)
        else:
            # Note that diabetic_idx will be ignored if idx_type = 'full'
            init_state = State(
                    state_idx=state_idx, idx_type=idx_type,
                    diabetic_idx=diabetic_idx)

        return init_state

    def generate_random_state(self, diabetic_idx=None):
        # Note that we will condition on diabetic idx if provided
        if diabetic_idx is None:
            diabetic_idx = np.random.binomial(1, self.p_diabetes)

        # hr and sys_bp w.p. [.25, .5, .25]
        hr_state = np.random.choice(np.arange(3), p=np.array([.25, .5, .25]))
        sysbp_state = np.random.choice(np.arange(3), p=np.array([.25, .5, .25]))
        # percoxyg w.p. [.2, .8]
        percoxyg_state = np.random.choice(np.arange(2), p=np.array([.2, .8]))

        if diabetic_idx == 0:
            glucose_state = np.random.choice(np.arange(5), \
                p=np.array([.05, .15, .6, .15, .05]))
        else:
            glucose_state = np.random.choice(np.arange(5), \
                p=np.array([.01, .05, .15, .6, .19]))
        antibiotic_state = 0
        vaso_state = 0
        vent_state = 0

        state_categs = [hr_state, sysbp_state, percoxyg_state,
                glucose_state, antibiotic_state, vaso_state, vent_state]

        return State(state_categs=state_categs, diabetic_idx=diabetic_idx)

    def transition_antibiotics_on(self):
        '''
        antibiotics state on
        heart rate, sys bp: hi -> normal w.p. .5
        '''
        self.state.antibiotic_state = 1
        if self.state.hr_state == 2 and np.random.uniform(0,1) < 0.5:
            self.state.hr_state = 1
        if self.state.sysbp_state == 2 and np.random.uniform(0,1) < 0.5:
            self.state.sysbp_state = 1

    def transition_antibiotics_off(self):
        '''
        antibiotics state off
        if antibiotics was on: heart rate, sys bp: normal -> hi w.p. .1
        '''
        if self.state.antibiotic_state == 1:
            if self.state.hr_state == 1 and np.random.uniform(0,1) < 0.1:
                self.state.hr_state = 2
            if self.state.sysbp_state == 1 and np.random.uniform(0,1) < 0.1:
                self.state.sysbp_state = 2
            self.state.antibiotic_state = 0

    def transition_vent_on(self):
        '''
        ventilation state on
        percent oxygen: low -> normal w.p. .7
        '''
        self.state.vent_state = 1
        if self.state.percoxyg_state == 0 and np.random.uniform(0,1) < 0.7:
            self.state.percoxyg_state = 1

    def transition_vent_off(self):
        '''
        ventilation state off
        if ventilation was on: percent oxygen: normal -> lo w.p. .1
        '''
        if self.state.vent_state == 1:
            if self.state.percoxyg_state == 1 and np.random.uniform(0,1) < 0.1:
                self.state.percoxyg_state = 0
            self.state.vent_state = 0

    def transition_vaso_on(self):
        '''
        vasopressor state on
        for non-diabetic:
            sys bp: low -> normal, normal -> hi w.p. .7
        for diabetic:
            raise blood pressure: normal -> hi w.p. .9,
                lo -> normal w.p. .5, lo -> hi w.p. .4
            raise blood glucose by 1 w.p. .5
        '''
        self.state.vaso_state = 1
        if self.state.diabetic_idx == 0:
            if np.random.uniform(0,1) < 0.7:
                if self.state.sysbp_state == 0:
                    self.state.sysbp_state = 1
                elif self.state.sysbp_state == 1:
                    self.state.sysbp_state = 2
        else:
            if self.state.sysbp_state == 1:
                if np.random.uniform(0,1) < 0.9:
                    self.state.sysbp_state = 2
            elif self.state.sysbp_state == 0:
                up_prob = np.random.uniform(0,1)
                if up_prob < 0.5:
                    self.state.sysbp_state = 1
                elif up_prob < 0.9:
                    self.state.sysbp_state = 2
            if np.random.uniform(0,1) < 0.5:
                self.state.glucose_state = min(4, self.state.glucose_state + 1)

    def transition_vaso_off(self):
        '''
        vasopressor state off
        if vasopressor was on:
            for non-diabetics, sys bp: normal -> low, hi -> normal w.p. .1
            for diabetics, blood pressure falls by 1 w.p. .05 instead of .1
        '''
        if self.state.vaso_state == 1:
            if self.state.diabetic_idx == 0:
                if np.random.uniform(0,1) < 0.1:
                    self.state.sysbp_state = max(0, self.state.sysbp_state - 1)
            else:
                if np.random.uniform(0,1) < 0.05:
                    self.state.sysbp_state = max(0, self.state.sysbp_state - 1)
            self.state.vaso_state = 0

    def transition_fluctuate(self, hr_fluctuate, sysbp_fluctuate, percoxyg_fluctuate, \
        glucose_fluctuate):
        '''
        all (non-treatment) states fluctuate +/- 1 w.p. .1
        exception: glucose flucuates +/- 1 w.p. .3 if diabetic
        '''
        if hr_fluctuate:
            hr_prob = np.random.uniform(0,1)
            if hr_prob < 0.1:
                self.state.hr_state = max(0, self.state.hr_state - 1)
            elif hr_prob < 0.2:
                self.state.hr_state = min(2, self.state.hr_state + 1)
        if sysbp_fluctuate:
            sysbp_prob = np.random.uniform(0,1)
            if sysbp_prob < 0.1:
                self.state.sysbp_state = max(0, self.state.sysbp_state - 1)
            elif sysbp_prob < 0.2:
                self.state.sysbp_state = min(2, self.state.sysbp_state + 1)
        if percoxyg_fluctuate:
            percoxyg_prob = np.random.uniform(0,1)
            if percoxyg_prob < 0.1:
                self.state.percoxyg_state = max(0, self.state.percoxyg_state - 1)
            elif percoxyg_prob < 0.2:
                self.state.percoxyg_state = min(1, self.state.percoxyg_state + 1)
        if glucose_fluctuate:
            glucose_prob = np.random.uniform(0,1)
            if self.state.diabetic_idx == 0:
                if glucose_prob < 0.1:
                    self.state.glucose_state = max(0, self.state.glucose_state - 1)
                elif glucose_prob < 0.2:
                    self.state.glucose_state = min(1, self.state.glucose_state + 1)
            else:
                if glucose_prob < 0.3:
                    self.state.glucose_state = max(0, self.state.glucose_state - 1)
                elif glucose_prob < 0.6:
                    self.state.glucose_state = min(4, self.state.glucose_state + 1)

    def calculateReward(self):
        num_abnormal = self.state.get_num_abnormal()
        if num_abnormal >= 3:
            return -1
        elif num_abnormal == 0 and not self.state.on_treatment():
            return 1
        return 0

    def transition(self, action):
        self.state = self.state.copy_state()

        if action.antibiotic == 1:
            self.transition_antibiotics_on()
            hr_fluctuate = False
            sysbp_fluctuate = False
        elif self.state.antibiotic_state == 1:
            self.transition_antibiotics_off()
            hr_fluctuate = False
            sysbp_fluctuate = False
        else:
            hr_fluctuate = True
            sysbp_fluctuate = True

        if action.ventilation == 1:
            self.transition_vent_on()
            percoxyg_fluctuate = False
        elif self.state.vent_state == 1:
            self.transition_vent_off()
            percoxyg_fluctuate = False
        else:
            percoxyg_fluctuate = True

        glucose_fluctuate = True

        if action.vasopressors == 1:
            self.transition_vaso_on()
            sysbp_fluctuate = False
            glucose_fluctuate = False
        elif self.state.vaso_state == 1:
            self.transition_vaso_off()
            sysbp_fluctuate = False

        self.transition_fluctuate(hr_fluctuate, sysbp_fluctuate, percoxyg_fluctuate, \
            glucose_fluctuate)

        return self.calculateReward()

    def select_actions(self):
        assert self.policy_array is not None
        probs = self.policy_array[
                    self.state.get_state_idx(self.policy_idx_type)
                ]
        aev_idx = np.random.choice(np.arange(Action.NUM_ACTIONS_TOTAL), p=probs)
        return Action(action_idx = aev_idx)
