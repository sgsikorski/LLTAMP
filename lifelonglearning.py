import util

# Sampling based policy
def InitialPolicy(state):
    action = {}
    action["Type"] = util.ACTION_TYPES[0]
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