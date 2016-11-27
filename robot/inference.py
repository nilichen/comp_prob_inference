#!/usr/bin/env python
# inference.py
# Base code by George H. Chen (georgehc@mit.edu) -- updated 10/18/2016
import collections
import sys

import graphics
import numpy as np
import robot


# Throughout the code, we use these variables.
# Do NOT change these (but you'll need to use them!):
# - all_possible_hidden_states: a list of possible hidden states
# - all_possible_observed_states: a list of possible observed states
# - prior_distribution: a distribution over states
# - transition_model: a function that takes a hidden state and returns a
#     Distribution for the next state
# - observation_model: a function that takes a hidden state and returns a
#     Distribution for the observation from that hidden state
all_possible_hidden_states = robot.get_all_hidden_states()
all_possible_observed_states = robot.get_all_observed_states()
prior_distribution = robot.initial_distribution()
transition_model = robot.transition_model
observation_model = robot.observation_model

M = len(all_possible_hidden_states)
transition_matrix = np.zeros((M, M))
for i, hs in enumerate(all_possible_hidden_states):
    for ns, v in transition_model(hs).items():
        transition_matrix[(i, all_possible_hidden_states.index(ns))] = v

N = len(all_possible_observed_states)
observation_matrix = np.zeros((M, N))
for i, hs in enumerate(all_possible_hidden_states):
    for ob, v in observation_model(hs).items():
        observation_matrix[i, all_possible_observed_states.index(ob)] = v

# You may find this function helpful for computing logs without yielding a
# NumPy warning when taking the log of 0.
def careful_log(x):
    # computes the log of a non-negative real number
    if x == 0:
        return -np.inf
    else:
        return np.log(x)
logv = np.vectorize(careful_log)

# -----------------------------------------------------------------------------
# Functions for you to implement
#

def forward_backward(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    robot.py and see how it is used in both robot.py and the function
    generate_data() above, and the i-th Distribution should correspond to time
    step i
    """


    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = np.zeros(M)
    for d, v in prior_distribution.items():
        forward_messages[0][all_possible_hidden_states.index(d)] = v

    backward_messages = [None] * num_time_steps
    backward_messages[0] = np.ones(M)

    for i in range(num_time_steps-1):
        observationF = observations[i]
        observationB = observations[num_time_steps-i-1]

        F = forward_messages[i] * (observation_matrix[:, all_possible_observed_states.index(observationF)] if observationF else 1)
        forward_messages[i+1] = transition_matrix.T.dot(F)

        B = backward_messages[i] * (observation_matrix[:, all_possible_observed_states.index(observationB)] if observationB else 1)
        backward_messages[i+1] = transition_matrix.dot(B)


    marginals = [None] * num_time_steps 
    for i in range(num_time_steps):
        observation = observations[i]
        marginal_dict = {}
        marginal_array = forward_messages[i] * backward_messages[num_time_steps-i-1] * \
        (observation_matrix[:, all_possible_observed_states.index(observation)] if observation else 1)
        for j, hs in enumerate(all_possible_hidden_states):
            p = marginal_array[j]
            if p != 0:
                marginal_dict[hs] = p
        marginals[i] = robot.Distribution(marginal_dict)
        marginals[i].renormalize()


    return marginals


def Viterbi(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """
    observation_matrix_log = logv(observation_matrix)
    transition_matrix_log = logv(transition_matrix)
    num_time_steps = len(observations)
    # messages = [None] * num_time_steps
    message = np.zeros(M)
    for d, v in prior_distribution.items():
        message[all_possible_hidden_states.index(d)] = v
    message = logv(message)  
    traceback_tables = [None] * num_time_steps

    for i in range(num_time_steps-1):
        observation = observations[i]
        # matrix = message * (observation_matrix[:, all_possible_observed_states.index(observation)] if observation else 1) + transition_matrix
        matrix = message + (observation_matrix_log[:, all_possible_observed_states.index(observation)] if observation else 0) + transition_matrix_log.T
        message = np.max(matrix, axis=1)
        # message = message / message.sum()
        traceback_tables[i+1] = np.argmax(matrix, axis=1)
    
    final_observation = observations[-1]
    final_hs = (message + (observation_matrix_log[:, all_possible_observed_states.index(final_observation)] if final_observation else 0)).argmax()

    estimated_hidden_states = [None] * num_time_steps 
    estimated_hidden_states[num_time_steps-1] = all_possible_hidden_states[final_hs]
    for i in range(num_time_steps-1):
        if i == 0:
            last_hs = final_hs
        hs = traceback_tables[num_time_steps-i-1][last_hs]
        last_hs = hs
        estimated_hidden_states[num_time_steps-i-2] = all_possible_hidden_states[hs]

    return estimated_hidden_states


def second_best(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #


    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps # remove this

    return estimated_hidden_states


# -----------------------------------------------------------------------------
# Generating data from the hidden Markov model
#

def generate_data(num_time_steps, make_some_observations_missing=False,
                  random_seed=None):
    # generate samples from this project's hidden Markov model
    hidden_states = []
    observations = []

    # if the random seed is not None, then this makes the randomness
    # deterministic, which may be helpful for debug purposes
    np.random.seed(random_seed)

    # draw initial state and emit an observation
    initial_state = prior_distribution.sample()
    initial_observation = observation_model(initial_state).sample()

    hidden_states.append(initial_state)
    observations.append(initial_observation)

    for time_step in range(1, num_time_steps):
        # move the robot
        prev_state = hidden_states[-1]
        new_state = transition_model(prev_state).sample()

        # maybe emit an observation
        if not make_some_observations_missing:
            new_observation = observation_model(new_state).sample()
        else:
            if np.random.rand() < .1:  # 0.1 prob. of observation being missing
                new_observation = None
            else:
                new_observation = observation_model(new_state).sample()

        hidden_states.append(new_state)
        observations.append(new_observation)

    return hidden_states, observations


# -----------------------------------------------------------------------------
# Main
#

def main():
    # flags
    make_some_observations_missing = False
    use_graphics = False
    need_to_generate_data = True

    # parse command line arguments
    for arg in sys.argv[1:]:
        if arg == '--missing':
            make_some_observations_missing = True
        elif arg == '--nographics':
            use_graphics = False
        elif arg.startswith('--load='):
            filename = arg[7:]
            hidden_states, observations = robot.load_data(filename)
            need_to_generate_data = False
            num_time_steps = len(hidden_states)

    # if no data is loaded, then generate new data
    if need_to_generate_data:
        num_time_steps = 100
        hidden_states, observations = \
            generate_data(num_time_steps,
                          make_some_observations_missing)

    print('Running forward-backward...')
    marginals = forward_backward(observations)
    print("\n")

    timestep = 99
    print("Most likely parts of marginal at time %d:" % (timestep))
    if marginals[timestep] is not None:
        print(sorted(marginals[timestep].items(),
                     key=lambda x: x[1],
                     reverse=True)[:10])
    else:
        print('*No marginal computed*')
    print("\n")

    print('Running Viterbi...')
    estimated_states = Viterbi(observations)
    print("\n")

    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states[time_step] is None:
            print('Missing')
        else:
            print(estimated_states[time_step])
    print("\n")

    print('Finding second-best MAP estimate...')
    estimated_states2 = second_best(observations)
    print("\n")

    print("Last 10 hidden states in the second-best MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states2[time_step] is None:
            print('Missing')
        else:
            print(estimated_states2[time_step])
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP estimate and true hidden " +
          "states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states2[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between second-best MAP estimate and " +
          "true hidden states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != estimated_states2[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP and second-best MAP " +
          "estimates:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    # display
    if use_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()


if __name__ == '__main__':
    main()
