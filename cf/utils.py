import numpy as np
import pandas as pd
from sepsisSimDiabetes.State import State

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib.ticker import FormatStrFormatter
warnings.simplefilter(action='ignore', category=FutureWarning)

def format_dgen_samps(states, actions, rewards, hidden, NSTEPS, NSIMSAMPS):
    """format_dgen_samps
    Formats the output of the data generator (a batch of trajectories) in a way
    that the other functions will consume

    :param states: states
    :param actions: actions 
    :param rewards: rewards
    :param hidden: hidden states
    :param NSTEPS: Maximum length of trajectory
    :param NSIMSAMPS: Number of trajectories
    """
    obs_samps = np.zeros((NSIMSAMPS, NSTEPS, 7))
    obs_samps[:, :, 0] = np.arange(NSTEPS)  # Time Index
    obs_samps[:, :, 1] = actions[:, :, 0]
    obs_samps[:, :, 2] = states[:, :-1, 0]  # from_states
    obs_samps[:, :, 3] = states[:, 1:, 0]  # to_states
    obs_samps[:, :, 4] = hidden[:, :, 0]  # Hidden variable
    obs_samps[:, :, 5] = hidden[:, :, 0]  # Hidden variable
    obs_samps[:, :, 6] = rewards[:, :, 0]

    return obs_samps

def df_from_samps(samps, pt_idx=0, get_outcome=False, is_proj=False):
    """df_from_samps

    Creates a dataframe from samples, selecting a specific patient in a batch,
    and formatting in a way that is consumed by our plotting code

    :param samps: Sample trajectories
    :param pt_idx: Patient index
    :param get_outcome: Boolean, whether or not to return the outcome
    :param is_proj: Whether or not this has been projected already
    """
    # Find the end of the trajectory, which is one past the time when reward occurs
    endtime = samps.shape[1] - 1 # By default, this is the end of the sequence
    for t in range(samps.shape[1]):
        if samps[pt_idx, t, 1] == -1:  # Action = -1 indicates end
            endtime = t
            break

    # Extract individual arrays
    if is_proj:
        # For projected samples, want one step back, b/c last state is abs
        time = np.arange(endtime)
    elif endtime == samps.shape[1] - 1:
        time = samps[pt_idx, :, 0].astype(int)
    else:
        time = np.arange(endtime + 1)  # +1 to get inclusive

    states = samps[pt_idx, time, 2]  # Go though endtime inclusive
    diab_idx = samps[pt_idx, 0, 4]  # Scalar

    state_array_2d = np.zeros((time.shape[0], 8))
    for t in time:
        state_array_2d[t, 0] = t
        if is_proj:
            if states[t] > 144:
                break
            this_state = State(
                    state_idx = states[t],
                    idx_type='proj_obs',
                    diabetic_idx=diab_idx)
        else:
            this_state = State(
                    state_idx = states[t],
                    idx_type='obs',
                    diabetic_idx=diab_idx)
        state_array_2d[t, 1:] = this_state.get_state_vector()

    df = pd.DataFrame(state_array_2d, columns = [
        'Time',
        'Heart Rate',
        'SysBP',
        'Percent O2',
        'Glucose',
        'Treat: AbX',
        'Treat: Vaso',
        'Treat: Vent'
    ])

    # Get the outcome
    if get_outcome and not is_proj:
        outcome = (endtime, samps[pt_idx, endtime - 1, 6])
        return df, outcome
    # The diff with proj is that the last state is at endtime-1
    elif get_outcome and is_proj:
        outcome = (endtime - 1, samps[pt_idx, endtime - 1, 6])
        return df, outcome
    else:
        return df

def plot_trajectory(samps, pt_idx=0, cf=False, cf_samps=None, cf_proj=False,
        max_plt_len=None, force_length=None):
    """plot_trajectory

    :param samps: Observed trajectory (output of format_dgen_samps)
    :param pt_idx: Patient Index
    :param cf: If true, plot distribution of counterfactuals
    :param cf_samps: If cf, then these are the cf samples
    :param cf_proj: Are these projected samples
    :param max_plt_len: Maximum length to plot
    :param force_length: Force length to a certain length
    """
    this_df, outcome = df_from_samps(samps, pt_idx, get_outcome=True)

    eps = 0.5
    param_dict = {
        'Heart Rate': {
            'ticks': ['Low', 'Normal', 'High'],
            'vals': [0, 1, 2],
            'nrange': [0.75, 1.25],
            'plt_outcome': True,
            'ylabel': 'HR'
        },
        'SysBP': {
            'ticks': ['Low', 'Normal', 'High'],
            'vals': [0, 1, 2],
            'nrange': [0.75, 1.25],
            'plt_outcome': True,
            'ylabel': 'SysBP'
        },
        'Percent O2': {
            'ticks': ['Low', 'Normal'],
            'vals': [0, 1],
            'nrange': [0.75, 1.25],
            'plt_outcome': True,
            'ylabel': 'Pct O2'
        },
        'Glucose': {
            'ticks': ['V. Low', 'Low', 'Normal', 'High', 'V. High'],
            'vals': [0, 1, 2, 3, 4],
            'nrange': [1.75, 2.25],
            'plt_outcome': True,
            'ylabel': 'Glucose'
        },
        'Treat: AbX': {
            'ticks': ['Off', 'On'],
            'vals': [0, 1],
            'nrange': None,
            'plt_outcome': True,
            'ylabel': 'Tx: Abx'
        },
        'Treat: Vaso':{
            'ticks': ['Off', 'On'],
            'vals': [0, 1],
            'nrange': None,
            'plt_outcome': True,
            'ylabel': 'Tx: Vaso'
        },
        'Treat: Vent': {
            'ticks': ['Off', 'On'],
            'vals': [0, 1],
            'nrange': None,
            'plt_outcome': True,
            'ylabel': 'Tx: Vent'
        },
    }

    outcome_symbol = {
            -1: {
                'marker': 'o',
                'color': 'r',
                'markersize': '10'
                },
            0: {
                'marker': 'o',
                'color': 'k',
                'markersize': '10'
                },
            1: {
                'marker': 'o',
                'color': 'g',
                'markersize': '10'
                }
            }

    fig, axes = plt.subplots(7, 1, sharex=True)
    fig.set_size_inches(8, 10)
    for i in range(7):
        this_col = this_df.columns[i+1]
        axes[i].plot(this_df['Time'], this_df[this_col], color='k')

        # Format the Y-axis according to the variable
        axes[i].set_ylabel(param_dict[this_col]['ylabel'])
        axes[i].set_yticks(param_dict[this_col]['vals'])
        axes[i].set_yticklabels(param_dict[this_col]['ticks'])
        axes[i].set_ylim(param_dict[this_col]['vals'][0] - eps,
                         param_dict[this_col]['vals'][-1]+ eps)

        # Plot the end of the sequence as red, green, black
        if param_dict[this_col]['plt_outcome']:
            obs_end_time = outcome[0]
            end_event = outcome[1].astype(int)
            axes[i].plot(
                obs_end_time,
                this_df[this_col][obs_end_time],
                marker=outcome_symbol[end_event]['marker'],
                color=outcome_symbol[end_event]['color']
                )

        nrange = param_dict[this_col]['nrange']
        last_time = this_df.shape[0]
        if force_length is not None:
            last_time = force_length
        if nrange is not None:
            axes[i].hlines(nrange, xmin=0, xmax=last_time,
                           colors='r',
                           linestyles='dotted', label='Normal Range')

        axes[i].set_xlim(-0.25, last_time + 0.5)
        # Format the X-axis as integers
        axes[i].xaxis.set_ticks(np.arange(0, last_time + 1, 2))
        axes[i].xaxis.set_major_formatter(FormatStrFormatter('%d'))

    if cf:
        if max_plt_len is None:
            max_plt_len = this_df.shape[0] + 1
        assert cf_samps is not None
        num_samps = cf_samps.shape[1]
        for i in range(num_samps):
            # import pdb
            # pdb.set_trace()
            this_df, outcome = \
                df_from_samps(cf_samps[:, i, :max_plt_len, :],
                              pt_idx, get_outcome=True, is_proj=cf_proj)
            for i in range(7):
                this_col = this_df.columns[i+1]
                # No CF trajectory for glucose
                if this_col == 'Glucose':
                    continue
                axes[i].plot(this_df['Time'], this_df[this_col], alpha=0.1, color='b')
                # Plot the end of the sequence as red, green, yellow
                if param_dict[this_col]['plt_outcome']:
                    end_time = outcome[0]
                    end_event = outcome[1].astype(int)
                    axes[i].plot(
                        end_time,
                        this_df[this_col][end_time],
                        marker=outcome_symbol[end_event]['marker'],
                        color=outcome_symbol[end_event]['color'],
                        alpha=0.3
                        )

    return fig, axes

