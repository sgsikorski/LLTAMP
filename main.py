from ai2thor.controller import Controller
import time
from util import State, Action, Transition, utilConstants
import lifelonglearning as ll
from tqdm import tqdm
import json
import argparse

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
    parser = argparse.ArgumentParser(description="Training and testing an interactive robot for task completion in iTHOR")
    parser.add_argument("-tr", "--train", action="store_true", help="Train the model")
    parser.add_argument("-te", "--test", action="store_true", help="Test the model")    
    parser.add_argument("-mp", "--model_path", dest='model_path', type=str, help="Path to the model")
    parser.add_argument("-gp", "--goal_path", dest='goal_path', default="goalTasks.json", type=str, help="Path to the goal tasks")
    parser.add_argument("-envs", "--environments", type=str, help="Environments to train and test the model on")
    parser.add_argument("-enum", "--enum", type=int, choices=range(1, 31), default=31, help="Number of environments to train and test the model on")
    args = parser.parse_args()

    model = ll.Model()
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

    # Add which environments we want to conduct over
    environments = []
    kitchens = [f"FloorPlan{i}" for i in range(1, args.enum)]
    living_rooms = [f"FloorPlan{200 + i}" for i in range(1, args.enum)]
    bedrooms = [f"FloorPlan{300 + i}" for i in range(1, args.enum)]
    bathrooms = [f"FloorPlan{400 + i}" for i in range(1, args.enum)]
    
    envsToUse = args.environments if args.environments is not None else "klbeba"
    if "k" in envsToUse: environemnts += kitchens
    if "l" in envsToUse: environemnts += living_rooms
    if "be" in envsToUse: environemnts += bedrooms
    if "ba" in envsToUse: environemnts += bathrooms

    try:
        with open(args.goal_path) as f:
            goalTasks = json.load(f)
    except FileNotFoundError as e:
        print(f"{e}. Please define a goal tasks .json file accordingly. 
              There is also an example goalTasks.json for use. Now terminating")
        return
    
    if args.train:
        train(controller, environments, goalTasks, model)
        model.saveModel(args.model_path)
    if args.test:
        model.loadModel(args.model_path)
        test(controller, environments, goalTasks, model)

    print("Done!")

if __name__ == '__main__':
    main()