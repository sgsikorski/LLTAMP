import time
from util import State, Action, Transition, utilConstants
import lifelonglearning as ll
from tqdm import tqdm

class Agent():
    def __init__(self, environments=[], goalTasks=[], model=None, verbose=False):
        self.Buffer = set()
        self.BufferStream = set()
        self.RRT = set()
        self.environments = environments
        self.goalTasks = goalTasks
        self.model = model

        self.verbose = verbose

        self.trainActionAmount = [0] * len(environments)
        self.testActionAmount = [0] * len(environments)

    # Training the interactive agent in the ai2thor environments
    def train(self, controller):
        for idx, environment in enumerate(tqdm(self.environments, desc="Training the model on the environments")):
            # Init this environment's buffer memory to empty
            M_k = set()
            RRT_k  = set()

            fails = 0

            # Get environment specific goal
            goal = self.goalTasks[idx]

            # Reset the controller to the new environment
            controller.reset(scene=environment)

            # Initial state
            state = State.State(controller.last_event.metadata, 
                                controller.step("GetReachablePositions").metadata["actionReturn"])
            M_k.add(state)

            # Run this environment actions for 100 seconds
            startTime = time.perf_counter()
            while(time.perf_counter() - startTime < 100):
                nTime = time.perf_counter()
                while (time.perf_counter() - nTime < 0.5):
                    continue
                # Determine action
                action = ll.InitialPolicy(state, goal)
                transition = Transition.Transition(state, action)
                RRT_k.add(transition)

                # Execute action
                event = controller.step(action.actionType, objectId=action.objectOn) if action.objectOn is not None else controller.step(action.actionType)

                if (not event.metadata["lastActionSuccess"]):
                    # Handle an unsuccessful action
                    # For now, we'll just try a new action
                    fails += 1
                    continue
            
                self.BufferStream.add(transition)

                # We accomplished the goal, do additional reporting here
                if (action.completeGoal):
                    print(f"Goal for {environment}_{idx+1} completed!")
                    controller.step(action="Done")
                    break
                
                newState = State.State(controller.last_event.metadata,
                                    controller.step("GetReachablePositions").metadata["actionReturn"])
                state = newState
                M_k.add(state)
            
            # Lifelong Learning Componenet
            self.model.update_theta(self.BufferStream, idx)

            # Shrink the buffer to size n - |M_k|
            self.Buffer = self.Buffer.union(M_k)
            self.RRT = self.RRT.union(RRT_k)
            self.trainActionAmount[idx] = len(M_k)

            if (self.verbose):
                print(f"The number of actions taken in environment {environment} is {len(M_k)}.")
                print(f"Goal: {goal}")
                print(f"State to achieve goal: {state}")
                print(f"The robot failed to do an action {fails} times.")
                for i, transition in enumerate(RRT_k):
                    print(f"Transition {i} : {transition}")
        return
            
    # Let's test the agent in the ai2thor environments
    def test(self, controller, timeLimit=100):
        successRate = 0
        for idx, environment in enumerate(tqdm(self.environments, desc="Testing the model on the environments")):
            numInitActions = 0
            numThetaActions = 0
            fails = 0
            transitionStream = set()
            controller.reset(scene=environment)

            # Initial state
            state = State.State(controller.last_event.metadata,
                                controller.step("GetReachablePositions").metadata["actionReturn"])

            goal = self.goalTasks[idx]
            
            startTime = time.perf_counter()
            while(time.perf_counter() - startTime < timeLimit):
                nTime = time.perf_counter()
                while (time.perf_counter() - nTime < 1):
                    continue
                # Determine action
                a0 = ll.InitialPolicy(state, goal)
                aHat = utilConstants.ALL_ACTIONS[ll.Model.predict(state)]
                
                # Pick action with better score
                action = min([a0, aHat], key=lambda a: ll.B(state, a, controller, goal))
                # Execute action
                event = controller.step(action.actionType, objectId=action.objectOn) if action.objectOn is not None else controller.step(action.actionType)
                if (not event.metadata["lastActionSuccess"]):
                    # Handle an unsuccessful action
                    # For now, we'll just try a new action
                    fails += 1
                    continue
            
                if (action == a0): numInitActions += 1
                if (action == aHat): numThetaActions += 1
                transition = Transition.Transition(state, action)
                transitionStream.add(transition)

                if action.completeGoal:
                    successRate += 1
                    print(f"Goal for {environment}_{idx+1} completed!")
                    controller.step(action="Done")
                    break
                
                newState = State.State(controller.last_event.metadata,
                                    controller.step("GetReachablePositions").metadata["actionReturn"])
                state = newState
            
            self.testActionAmount[idx] = numInitActions + numThetaActions

            if (self.verbose):
                print(f"The number of actions taken in environment {environment} is {numInitActions + numThetaActions}.")
                print(f"The robot failed to do an action {fails} times.")
                print(f"The number of actions taken from the initial policy is {numInitActions}.")
                print(f"The number of actions taken from the theta policy is {numThetaActions}.")
                for i, transition in enumerate(transitionStream):
                    print(f"Transition {i} : {transition}")

        print(f"There was a {successRate / len(self.environments)} success rate.")
        return
