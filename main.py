from ai2thor.controller import Controller
import time
import torch
from util import State, Action, Transition, utilConstants
import lifelonglearning as ll
from tqdm import tqdm
import json

# Some global variables
Buffer = set()
BufferStream = set()

# Training the interactive agent in the ai2thor environments
def train(controller, environments, goal):
    for environment in tqdm(environments, desc="Training the model on the environments"):
        # Init this environment's buffer memory to empty
        M_k = set()
        # Reset the controller to the new environment
        controller.reset(scene=environment)
        # event = controller.step("MoveAhead")

        # Initial state
        state = State.State(controller.last_event.metadata, 
                            controller.step("GetReachablePositions").metadata["actionReturn"])
        M_k.add(state)

        #controller.step("RotateLeft")
        #controller.step("RotateLeft")
        #controller.step("MoveBack")
        #controller.step("MoveBack")
        #controller.step("MoveBack")
        #s = (controller.last_event.metadata)

        # Run this environment actions for 100 seconds
        startTime = time.perf_counter()
        while(time.perf_counter() - startTime < 100):
            # Determine action
            action = ll.InitialPolicy(state, goal)
            transition = Transition.Transition(state, action)

            # Execute action
            if action.objectOn is not None:
                event = controller.step(action.actionType, objectId=action.objectOn)
            else:
                event = controller.step(action.actionType)
            if (action.completeGoal):
                break
            if (not event.metadata["lastActionSuccess"]):
                # Handle an unsuccessful action
                # For now, we'll just try a new action
                continue
            
            BufferStream.add(transition)

            # Select s_p and a_p from BufferStream
            # if (ll.D(transition) < ll.threshold):
            #     newTransition = ll.A_correct(state, BufferStream)
            #     M_k.append(newTransition)

            newState = State.State(controller.last_event.metadata,
                                   controller.step("GetReachablePositions").metadata["actionReturn"])
            state = newState
            M_k.add(state)
        
        # Lifelong Learning Componenet
        #theta_k = ll.A_cl()

        # Shrink the buffer to size n - |M_k|
        # Buffer = Buffer.union(M_k)
        print(f"The number of actions taken in this environment is {len(M_k)}")
        
# Let's test the agent in the ai2thor environments
def test(controller, environments, goalTask):
    successRate = 0
    for environment in tqdm(environments, desc="Testing the model on the environments"):
        numActions = 0
        controller.reset(scene=environment)

        # Initial state
        state = State.State(controller.last_event.metadata)
        
        startTime = time.perf_counter()
        while(time.perf_counter() - startTime < 100):
            # Determine action
            a0 = ll.InitialPolicy(state, goalTask["objectId"])
            aHat = None # ll.ThetaPolicy(state, a0, theta_k)
            
            # Pick action with better score
            action = max([a0, aHat], key=lambda a: ll.B(state, a))
            event = controller.step(action.actionType)
            if (not event.metadata["lastActionSuccess"]):
                # Handle an unsuccessful action
                # For now, we'll just try a new action
                continue
            numActions += 1
    return successRate

def main():
    controller = Controller(
        agentMode = "default",
        visibilityDistance=1.5,
        scene="FloorPlan1",
        gridSize=utilConstants.GRIDSIZE,
        movementGaussianSigma = 0,
        rotateStepDegress=90,
        rotateGaussianSigma = 0,
        fieldOfView = 90
    )
    # Add all iThor scenes
    kitchens = [f"FloorPlan{i}" for i in range(1, 31)]
    living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 31)]
    bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 31)]
    bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 31)]

    environments = kitchens + living_rooms + bedrooms + bathrooms

    try:
        with open("goalTasks.json") as f:
            goalTasks = json.load(f)
    except FileNotFoundError as e:
        print(f"{e}. Now terminating")
        return
    train(controller, environments, goalTasks[0])
    test(controller, environments, goalTasks[0])

    return

if __name__ == '__main__':
    main()