#!/usr/bin/env python

# Python imports.
import sys
import logging

# Other imports.
from simple_rl.agents import LinearQAgent
from simple_rl.tasks import GymMDP
from simple_rl.run_experiments import run_agents_on_mdp

def main(open_plot=True):
    # Gym MDP
    gym_mdp = GymMDP(env_name='CartPole-v0', render=True)
    num_feats = gym_mdp.get_num_state_feats()

    # Setup agents and run.
    q_learning_agent = LinearQAgent(gym_mdp.get_actions(), num_feats)
    run_agents_on_mdp([q_learning_agent], gym_mdp, instances=1, episodes=400, steps=210, open_plot=open_plot, verbose=True)

if __name__ == "__main__":
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.ERROR)
    main()