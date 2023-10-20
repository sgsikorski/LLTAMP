from util import utilConstants, Action, Node
import torch
import random
import numpy as np

# Threshold for learning. Will need to be fine tuned
threshold = 0.05

def findNearestNode(space, x, y):
    min = "inf"
    nearestNode = None
    for node in space:
        diff = np.sqrt(node.x - x)**2 + (node.y - y)**2
        if (diff) < min:
            min = diff
            nearestNode = node
    return nearestNode

# Decide to move in x or y direction based on which is closer to the sampled node
def useXorY(x1, y1, x2, y2):
    if (x1 - x2 == 0): return "y"
    if (y1 - y2 == 0): return "x"
    if (np.abs(x2 - x1) == np.abs(y2 - y1)):
        return ("x" if random.randint(0, 1) == 0 else "y")
    return ("x" if np.abs(x2 - x1) < np.abs(y2 - y1) else "y")
    

# Sampling based policy
def InitialPolicy(controller, state, space, goalObject):
    action = Action()
    potentialActions = utilConstants.getPotentialActions(state)

    # Do a 360 degree scan of the scene
    # If goal object is visible, move towards it
        # Will introduce limitations of object that obstruct movement but not vision
    obj = state.scanForGoal(controller, goalObject)
    if (obj is not None):
        # Move towards goal object
        objPos = obj["position"]
        useXorY(state.agentX, state.agentY, objPos["x"], objPos["y"])


    else:
    # Check if there's a grabable object in the scene
    # Allow sampling of actions on the grabable object
    # If multiple grabable objects, sample from the closest one

    # RRT to goal node at x=1, y=1
    # Randomly sample x, y discretized by .25 from 10 <= x, y <= 10
        x = random.randint(-40, 40) * utilConstants.GRIDSIZE
        y = random.randint(-40, 40) * utilConstants.GRIDSIZE

        magX = np.sign(x - state.agentX)
        magY = np.sign(y - state.agentY)

        # We'll choose to go the x or y direction by whichever is less to the sampled node
        nearestNode = findNearestNode(space, x, y)
        if (nearestNode is None):
            raise ValueError("Did you possibly not intialize the state space?")
        
        # Can one of the potential actions on the visible actions get us closer to x, y?

        
        if useXorY(x, y, nearestNode.x, nearestNode.y) == "x": magY = 0 
        else: magX = 0
        newNode = Node(nearestNode.x + magX * utilConstants.GRIDSIZE, 
                    nearestNode.y + magY * utilConstants.GRIDSIZE, 
                    nearestNode)
        space.append(newNode)

    action.actionType = ("MoveRight" if magX > 0 else "MoveLeft") if useXorY(x, y) == "x" else ("MoveAhead" if magY > 0 else "MoveBack")
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
    diff += state1.getNumOfDifferentVisibleObjects(state2)
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