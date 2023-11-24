from util import utilConstants, Action, State
import torch
from GradientEpisodicMemory.model.gem import Net
import random
import numpy as np

# Threshold for learning. Will need to be fine tuned
threshold = 0.05

class NetInputs():
    def __init__(self, n_layers, n_hiddens, lr, cuda, n_memories, memory_strength):
        self.n_layers = n_layers
        self.n_hiddens = n_hiddens
        self.lr = lr
        self.cuda = cuda
        self.n_memories = n_memories
        self.memory_strength = memory_strength
        self.data_file = None

# pi_theta policy model
class Model(Net):
    def __init__(self, input_dim=3, n_tasks=30):
        args = NetInputs(n_layers=2, 
                         n_hiddens=100, 
                         lr=0.001, 
                         cuda=False, 
                         n_memories=300, 
                         memory_strength=0.5)
        # Update input_dim to map from state to features
        super(Model, self).__init__(n_inputs=input_dim, 
                                    n_outputs=len(utilConstants.ALL_ACTIONS), 
                                    n_tasks=n_tasks, 
                                    args=args)
    
    def getThetaParameter(self):
        return self.parameters()

    def predict(self, state):
        with torch.no_grad():
            return self.net(state)
    
    def update_theta(self, Buffer, t):
        # Size of s needs to equals the input_dim of the model
        s = []
        a = []
        for transition in Buffer:
            s.append([transition.state.agentX, transition.state.agentY, transition.state.agentZ])
            a.append(utilConstants.ALL_ACTIONS.index(transition.action.actionType))
        for i in range(300 - len(s)):
            s.append([0, 0, 0])
            a.append(0)
        self.observe(torch.tensor(s), t, torch.tensor(a))

    def saveModel(self, path='models/trained_agent.pt'):
        torch.save(self.net.state_dict(), path)
    
    def loadModel(self, path='models/trained_agent.pt'):
        self.net.load_state_dict(torch.load(path))

# Sampling based policy
def InitialPolicy(state, goalTasks):
    objOn = None
    completeGoal = False
    # If goal object is visible, interact with it
        # Will introduce limitations of object that obstruct movement but not vision
    if (goalTasks["objectId"] in state.reachObjName):
        ob = [o for o in state.reachableObjects if o['isPickedUp']]
        if len(ob) > 0:
            actionType = 'DropHandObject'
            objOn = ob[0]['objectId']
        else:    
            objOn = goalTasks["objectId"]
            objOnProp = next((obj for obj in state.reachableObjects if obj["objectId"] == objOn), None)
            status = goalTasks["status"]
            if status == "Move":
                status = "MoveHeld" if objOnProp["pickupable"] else "Push"
            actionType = utilConstants.determineAction(status)
            completeGoal = True
    else:
        choices = 1 if len(state.reachableObjects) < 1 else 2
        match(random.randint(0, choices)):
            case 0:
                # Randomly sample x, y discretized by .25 from 10 <= x, y <= 10
                sam = random.randint(-40, 40) * utilConstants.GRIDSIZE
                samMag = np.sign(sam - state.agentX)
                actionType = ("MoveAhead" if samMag > 0 else "MoveBack")
            case 1:
                actionType = "RotateRight" if random.randint(0, 1) == 0 else "RotateLeft"
            case 2:
                # Random object to interact with
                objOn = random.choice(state.reachObjName)
                objOnProp = next((obj for obj in state.reachableObjects if obj["objectId"] == objOn), None)
                potentialActions = utilConstants.getPotentialActions(objOnProp)
                actionType = random.choice(utilConstants.MOVEMENT_ACTION_TYPES) if len(potentialActions) < 1 else random.choice(potentialActions)
                if actionType == 'MoveHeldObject' or actionType in utilConstants.MOVEMENT_ACTION_TYPES:
                    objOn = None
    action = Action.Action(actionType, objOn, completeGoal)
    return action

# We can define difference by measuring distance of agent locations
# objects in the scene, and the states of the object
# How different is state1 to state2
def difference(state1, state2):
    diff = 0
    # Take manhattan distance
    diff += state1.getManhattanDistance() - state2.getManhattanDistance()
    diff += state1.getObjDiff(state2)
    return diff

def B(s, a, controller, goal):
    b = "inf"
    if a.actionType in utilConstants.MOVEMENT_ACTION_TYPES:
        event = controller.step(a.actionType)
        if (not event.metadata["lastActionSuccess"]):
            return b
        b = difference(s, State.State(controller.last_event.metadata, 
                                    controller.step("GetReachablePositions").metadata["actionReturn"]))
        controller.step(a.getOppositeActionType())
    else:
        # Logic on how to judge if an action is a good idea
        if (goal["status"] == "Move"):
            objPos = goal["objPosition"]
            return np.sqrt((s.agentX - objPos["x"])**2 + (s.agentY - objPos["y"])**2 + (s.agentZ - objPos["z"])**2)
        if (goal["objectId"] in s.reachObjName):
            return 0
    return b