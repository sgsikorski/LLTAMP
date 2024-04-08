import time
import torch
import torch.nn.functional as F
from util import State, Action, Transition
from util import utilConstants as uc
import lifelonglearning as ll
from tqdm import tqdm
import random
import math

# ===================================
#  Hyperparameters
# ===================================
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.003
EPS_DECAY = 1000
TAU = 0.002
LR = 1e-4
EPISODE_LENGTH = 1000
# ==================================

class Agent():
    def __init__(self, args, environments=[], goalTasks=[], model=None):
        self.environments = environments
        self.goalTasks = goalTasks
        self.model = model

        self.timeout = args.timeout
        self.verbose = args.verbose
        self.epochs = args.epochs

        self.trainActionAmount = [0] * len(environments)
        self.testActionAmount = [0] * len(environments)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = ll.DQN(3+len(uc.OBJECT_TYPES), len(uc.ALL_ACTIONS)).to(self.device)
        self.target_net = ll.DQN(3+len(uc.OBJECT_TYPES), len(uc.ALL_ACTIONS)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LR, amsgrad=True)
    
    def goalCompleted(self, action, goal) -> bool:
        return (action.actionType == uc.determineAction(goal['status']) and action.objectOn == goal['objectId'])

    # Training the interactive agent in the ai2thor environments
    def train(self, controller):
        for epoch in range(self.epochs):
            for idx, environment in enumerate(tqdm(self.environments, desc=f"Going through each environment for epoch {epoch+1}")):
                # Get environment specific goal
                goal = self.goalTasks[idx]

                fails = 0

                # Reset the controller to the new environment
                controller.reset(scene=environment)
                controller.step(action="Initialize")
                controller.step(action="MoveAhead")
                controller.step(action="MoveLeft")

                # Initial state
                state = State.State(controller.last_event.metadata, 
                                    controller.step("GetReachablePositions").metadata["actionReturn"])
                
                episode = 0
                while(episode < EPISODE_LENGTH):
                    eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * episode / EPS_DECAY)

                    act = 0
                    # Determine action
                    if random.random() > eps:
                        # TODO: Elegant way to handle this
                        act = self.policy_net.maxQ(torch.tensor(state.getOneHotEncoding(), device=self.device))
                        if (uc.ALL_ACTIONS[act] not in state.getPossibleActions()):
                            act = uc.ALL_ACTIONS.index(random.sample(state.getPossibleActions(), 1)[0])
                    else:
                        act = uc.ALL_ACTIONS.index(random.sample(state.getPossibleActions(), 1)[0])
                    
                    action = Action.Action(uc.ALL_ACTIONS[act], state.chooseFromReach(uc.ALL_ACTIONS[act]), False)
                    print(f"Choosing Action: {action}")
                    # Execute action
                    if action.objectOn is not None:
                        event = controller.step(action.actionType, objectId=action.objectOn) 
                    else:
                        event = controller.step(action.actionType)

                    newState = State.State(controller.last_event.metadata,
                                        controller.step("GetReachablePositions").metadata["actionReturn"])
                    reward = newState.getReward(goal)

                    if (not event.metadata["lastActionSuccess"]):
                        # Handle an unsuccessful action
                        # For now, we'll just try a new action
                        print(f"Action {action} failed.")
                        fails += 1
                        # continue
                    
                    # We accomplished the goal, do additional reporting here
                    if (self.goalCompleted(action, goal)):
                        print(f"Goal for {environment}_{idx+1} completed!")

                        state_action_values = self.policy_net(torch.tensor(state.getOneHotEncoding(), device=self.device))[act]
                        expected_state_action_values = torch.tensor(1.0)
                        loss = F.mse_loss(state_action_values, expected_state_action_values)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        break
                    
                    state_action_values = self.policy_net(torch.tensor(state.getOneHotEncoding(), device=self.device))[act]
                    expected_state_action_values = reward + GAMMA * torch.max(self.target_net(torch.tensor(state.getOneHotEncoding(), device=self.device)))
                    loss = F.mse_loss(state_action_values, expected_state_action_values)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if idx % 64 == 0:
                        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                            target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)
                    
                    # Create the new state and add it to the memory
                    newState = State.State(controller.last_event.metadata,
                                        controller.step("GetReachablePositions").metadata["actionReturn"])
                    state = newState
                    episode += 1
                
                controller.step(action="Done")
                
                # TODO: Lifelong Learning Componenet
                # self.model.update_theta(self.BufferStream, idx)

                if (self.verbose):
                    print(f"Goal: {goal}")
                    print(f"State to achieve goal: {state}")
                    print(f"The robot failed to do an action {fails} times.")
            print(f"Epoch {epoch+1} completed!")
        return
    
    
            
    # Let's test the agent in the ai2thor environments
    def test(self, controller):
        # Allow screen recording to be set up
        stime = time.perf_counter()
        while(time.perf_counter() - stime < 10):
            continue
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
            while(time.perf_counter() - startTime < self.timeout):
                nTime = time.perf_counter()
                while (time.perf_counter() - nTime < 0.5):
                    continue
                # Determine action
                a0 = ll.InitialPolicy(state, goal)
                aHatType = uc.ALL_ACTIONS[self.model.predict(
                    torch.tensor([[state.agentX, state.agentY, state.agentZ]]))[0].argmax()]
                if (aHatType == "TASKDONE"):
                    aHat = Action.Action("Done", None, True)
                else:
                    if aHatType in uc.MOVEMENT_ACTION_TYPES:
                        aHat = Action.Action(aHatType, None, False)
                    else:
                        aHat = Action.Action(aHatType, state.chooseFromReach(aHatType), False)
            
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
            controller.step(action="Done")
            if (self.verbose):
                print(f"The number of actions taken in environment {environment} is {numInitActions + numThetaActions}.")
                print(f"The robot failed to do an action {fails} times.")
                print(f"The number of actions taken from the initial policy is {numInitActions}.")
                print(f"The number of actions taken from the theta policy is {numThetaActions}.")
                for i, transition in enumerate(transitionStream):
                    print(f"Transition {i} : {transition}")

        print(f"There was a {successRate / len(self.environments)} success rate.")
        return
