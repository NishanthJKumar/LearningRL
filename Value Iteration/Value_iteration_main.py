# The deepcopy function is important to prevent shallow copying when updating the utility vector
from copy import deepcopy

actions_list = ["north", "south", "east", "west"] # The list of possible actions in the gridworld
# The list of legal states that is specific to this gridworld
states_list = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3),
               (4, 1), (4, 2), (4, 3)]
goal_state = (4, 3)
lava_state = (4, 2)

# This function takes in a state and an action and outputs the resulting state.
# Allows transition to illegal states (through walls). This is necessary for the transition_probability
# function to work correctly


def action_to_state(action, current_state):

    new_state = (0, 0)
    if (action == "north"):
        new_state = (current_state[0], current_state[1] + 1)
    elif (action == "south"):
        new_state = (current_state[0], current_state[1] - 1)
    elif (action == "east"):
        new_state = (current_state[0] + 1, current_state[1])
    elif (action == "west"):
        new_state = (current_state[0] - 1, current_state[1])
    else:
        raise ValueError("The agent was told to take an invalid action")

    return new_state

# This function takes in a current state, an action and a new state and returns the probability that the agent will
# arrive in the new state given that it takes the specified action from the current state


def transition_probability(curr_state, new_state, action):

    # If the agent is in one of the terminal states, then it can't leave
    if curr_state == goal_state or curr_state == lava_state:
        return 0.0
    elif action_to_state(action, curr_state) == new_state:
        return 0.7
    else:
        return 0.1

# Reward function for the agent


def reward(new_state):
    if new_state == goal_state:
        return 1.0
    elif new_state == lava_state:
        return -1.0
    else:
        return -0.04

# This function actually performs value iteration given a state space, a list of actions and a limit of error
# (delta) at which to stop. At every iteration, it prints the utility vector for the states


def calculate_value_function(state_space, action_list, util_error, gamma):
    utility_vector = [[0, 0, 0],
                      [0, None, 0],
                      [0, 0, 0],
                      [0, -1, 1]]
    utility_vector_prime = [[0, 0, 0],
                            [0, None, 0],
                            [0, 0, 0],
                            [0, -1, 1]]
    # Initialize delta large enough that the program goes into the while loop below
    delta = 100

    while (delta > util_error * (1 - gamma)) / gamma:
        print(utility_vector_prime)

        # If deepcopy is not used, utility_vector and utility_vector_prime will always be equal
        utility_vector = deepcopy(utility_vector_prime)
        delta = 0.0

        for state in state_space:
            new_states_list = []
            action_vector_values = []

            # This for loop generates a list of all possible states that can be reached from this state
            for action in action_list:
                new_states_list.append(action_to_state(action, state))

            # This loops through all actions and all reachable states to calculate expected utilities
            for act in action_list:
                values_for_states = []
                for prime_state in new_states_list:
                    # The below condition is necessary because the action_to_state function returns illegal states
                    # Therefore, if an illegal state is returned, then we can assume the agent just stays in its
                    # current state. However, we still need to pass the illegal state into the transition
                    # function.
                    if prime_state in states_list:
                        # The prime_state[x] - 1 appears because the states begin with (1,1), but lists in python
                        # are 0 indexed (so the bottom left corner state would be (0,0)
                        values_for_states.append(transition_probability(state, prime_state, act) *
                                                 (utility_vector[prime_state[0] - 1][prime_state[1] - 1]))

                    else:
                        values_for_states.append(transition_probability(state, prime_state, act) *
                                                 (utility_vector[state[0] - 1][state[1] - 1]))

                action_vector_values.append(sum(values_for_states))

            # Bellman Update
            utility_vector_prime[state[0] - 1][state[1] - 1] = reward(state) + gamma * (max(action_vector_values))

            if(abs(utility_vector_prime[state[0] - 1][state[1] - 1] - utility_vector[state[0] - 1][state[1] - 1]) > delta):
                delta = abs(utility_vector_prime[state[0] - 1][state[1] - 1] - utility_vector[state[0] - 1][state[1] - 1])


# Run Value Iteration on the 4*3 gridworld from the Russel and Norvig AI textbook with transition dynamics changed
# (i.e, 0.7 probability agent moves in given direction and 0.1 probability it accidentally takes any of the three other
# actions)
calculate_value_function(states_list, actions_list, 0.01, 1.0)