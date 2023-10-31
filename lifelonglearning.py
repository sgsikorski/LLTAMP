from util import utilConstants, Action, Node
import torch
import random
import numpy as np

# Threshold for learning. Will need to be fine tuned
threshold = 0.05

# Sampling based policy
def InitialPolicy(state, goalTasks):
    objOn = None
    # If goal object is visible, interact with it
        # Will introduce limitations of object that obstruct movement but not vision
    if (goalTasks["objectId"] in state.reachableObjects):
        objOn = goalTasks["objectId"]
        actionType = utilConstants.determineAction(goalTasks["status"])
    else:
        match(random.randint(0, 2)):
            case 0:
                # Randomly sample x, y discretized by .25 from 10 <= x, y <= 10
                sam = random.randint(-40, 40) * utilConstants.GRIDSIZE
                samMag = np.sign(sam - state.agentY)
                actionType = ("MoveAhead" if samMag > 0 else "MoveBack")
            case 1:
                actionType = "RotateRight" if random.randint(0, 1) == 0 else "RotateLeft"
            case 2:
                # Change to randomly pick action of possible actions
                actionType = "PickupObject"
    action = Action(actionType, objOn)
    return action

# Learnable policy
def ThetaPolicy():

    return

# We can define difference by measuring distance of agent locations
# objects in the scene, and the states of the object
# How different is state1 to state2
def difference(state1, state2):
    diff = 0
    # Take manhattan distance
    diff += state1.getManhattanDistance() - state2.getManhattanDistance()
    diff += state1.getObjDiff(state2)
    return diff

# Checking if a transition (state, action) is suboptimal
# This is essentially using the learnable policy if the regular policy
# can't find a good action and recovers/backtracks
def D(state, action):
    # Scoring method of a state and action to get to goal
    # Scored on how much closer we are to the goal object and if a non-movement
    # action satisfies the goal state of goal object or reduces the
    # distance between current state and goal state by moving an object
    return 0

# From a transition in the bStream, find the least different state to the
# passed in state such that D >= threshold
def A_correct(state_p, bStream):
    candiates = []
    for transition in bStream:
        if (D(transition.state, transition.action) >= threshold):
            candiates.append(transition.state)
    return min(candiates, key=lambda s: difference(s, state_p))