import util
import torch
import tensorflow as tf

# Sampling based policy
def InitialPolicy(state):
    action = util.Action()

    # Check if there's a grabable object in the scene
    # Allow sampling of actions on the grabable object
    # If multiple grabable objects, sample from the closest one
    return action

# Learnable policy
def ThetaPolicy():

    return

# Checking if a transition is suboptimal
# This is essentially using the learnable policy if the regular policy
# can't find a good action and recovers/backtracks
def D(transition):

    return 0

# From a transition in the bStream, find the most similar state to the
# passed in state such that D >= threshold
def A_correct(state, bStream):

    return