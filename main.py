from ai2thor.controller import Controller
import time
from util import State, Action, Transition, utilConstants
import lifelonglearning as ll
from tqdm import tqdm
import json

# Some global variables
Buffer = set()
BufferStream = set()
RRT = set()

# Training the interactive agent in the ai2thor environments
def train(controller, environments, goalTasks, model):
    for idx, environment in enumerate(tqdm(environments, desc="Training the model on the environments")):
        # Init this environment's buffer memory to empty
        M_k = set()
        RRT_k  = set()

        # Get environment specific goal
        goal = goalTasks[idx]

        # Reset the controller to the new environment
        controller.reset(scene=environment)

        # Initial state
        state = State.State(controller.last_event.metadata, 
                            controller.step("GetReachablePositions").metadata["actionReturn"])
        M_k.add(state)

        # Run this environment actions for 100 seconds
        startTime = time.perf_counter()
        while(time.perf_counter() - startTime < 100):
            # Determine action
            action = ll.InitialPolicy(state, goal)
            transition = Transition.Transition(state, action)
            RRT_k.add(transition)

            # Execute action
            if action.objectOn is not None:
                event = controller.step(action.actionType, objectId=action.objectOn)
            else:
                event = controller.step(action.actionType)

            # We accomplished the goal, do additional work here
            if (action.completeGoal):
                break
            if (not event.metadata["lastActionSuccess"]):
                # Handle an unsuccessful action
                # For now, we'll just try a new action
                continue
            
            BufferStream.add(transition)
            newState = State.State(controller.last_event.metadata,
                                   controller.step("GetReachablePositions").metadata["actionReturn"])
            state = newState
            M_k.add(state)
        
        # Lifelong Learning Componenet
        model.update_theta(M_k, Buffer)

        # Shrink the buffer to size n - |M_k|
        # Buffer = Buffer.union(M_k)
        RRT = RRT.union(RRT_k)
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

    environments = kitchens # + living_rooms + bedrooms + bathrooms

    try:
        with open("goalTasks.json") as f:
            goalTasks = json.load(f)
    except FileNotFoundError as e:
        print(f"{e}. Now terminating")
        return
    
    model = ll.Model()
    train(controller, environments, goalTasks, model)
    test(controller, environments, goalTasks, model)

    return

if __name__ == '__main__':
    main()