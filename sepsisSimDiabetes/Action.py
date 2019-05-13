import numpy as np

class Action(object):

    NUM_ACTIONS_TOTAL = 8
    ANTIBIOTIC_STRING = "antibiotic"
    VENT_STRING = "ventilation"
    VASO_STRING = "vasopressors"
    ACTION_VEC_SIZE = 3

    def __init__(self, selected_actions = None, action_idx = None):
        assert (selected_actions is not None and action_idx is None) \
            or (selected_actions is None and action_idx is not None), \
            "must specify either set of action strings or action index"
        if selected_actions is not None:
            if Action.ANTIBIOTIC_STRING in selected_actions:
                self.antibiotic = 1
            else:
                self.antibiotic = 0
            if Action.VENT_STRING in selected_actions:
                self.ventilation = 1
            else:
                self.ventilation = 0
            if Action.VASO_STRING in selected_actions:
                self.vasopressors = 1
            else:
                self.vasopressors = 0
        else:
            mod_idx = action_idx
            term_base = Action.NUM_ACTIONS_TOTAL/2
            self.antibiotic = np.floor(mod_idx/term_base).astype(int)
            mod_idx %= term_base
            term_base /= 2
            self.ventilation = np.floor(mod_idx/term_base).astype(int)
            mod_idx %= term_base
            term_base /= 2
            self.vasopressors = np.floor(mod_idx/term_base).astype(int)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
            self.antibiotic == other.antibiotic and \
            self.ventilation == other.ventilation and \
            self.vasopressors == other.vasopressors

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_action_idx(self):
        assert self.antibiotic in (0, 1)
        assert self.ventilation in (0, 1)
        assert self.vasopressors in (0, 1)
        return 4*self.antibiotic + 2*self.ventilation + self.vasopressors

    def __hash__(self):
        return self.get_action_idx()

    def get_selected_actions(self):
        selected_actions = set()
        if self.antibiotic == 1:
            selected_actions.add(Action.ANTIBIOTIC_STRING)
        if self.ventilation == 1:
            selected_actions.add(Action.VENT_STRING)
        if self.vasopressors == 1:
            selected_actions.add(Action.VASO_STRING)
        return selected_actions

    def get_abbrev_string(self):
        '''
        AEV: antibiotics, ventilation, vasopressors
        '''
        output_str = ''
        if self.antibiotic == 1:
            output_str += 'A'
        if self.ventilation == 1:
            output_str += 'E'
        if self.vasopressors == 1:
            output_str += 'V'
        return output_str

    def get_action_vec(self):
        return np.array([[self.antibiotic], [self.ventilation], [self.vasopressors]])
