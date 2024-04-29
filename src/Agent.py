import time
import torch
import torch.nn.functional as F
from util import State, Action, Transition
from util import utilConstants as uc
import lifelonglearning as ll
import ReplayBuffer
from tqdm import tqdm
import random
import math

import matplotlib.pyplot as plt
import imageio
import numpy as np

# ===================================
#  Hyperparameters
# ===================================
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.003
TAU = 0.002
LR = 1e-4
EPISODE_LENGTH = 500
EPS_DECAY = 1000
# ==================================

class Agent():
    def __init__(self, args, environments=[], goalTasks=[], model=None):
        self.environments = environments
        self.goalTasks = goalTasks
        self.model = model

        self.timeout = args.timeout
        self.verbose = args.verbose
        self.epochs = args.epochs

        self.memory = ReplayBuffer.ReplayBuffer(10000)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = ll.DQN(3+len(uc.OBJECT_TYPES)*(len(uc.OBJECT_PROPERTIES)+1), len(uc.ALL_ACTIONS)).to(self.device)
        self.target_net = ll.DQN(3+len(uc.OBJECT_TYPES)*(len(uc.OBJECT_PROPERTIES)+1), len(uc.ALL_ACTIONS)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LR, amsgrad=True)
    
    def saveModel(self, mp, envidx):
        torch.save(self.policy_net.state_dict(), f"{mp}_{envidx}_policy.pt")
        torch.save(self.target_net.state_dict(), f"{mp}_{envidx}_target.pt")

    def goalCompleted(self, action, goal) -> bool:
        return (action.actionType == uc.determineAction(goal['status']) and action.objectOn == goal['objectId'])
    
    def plotReward(self, rewards, avgReward, episodes, plotTitle="Rewards", xLabel="Episodes", yLabel="Rewards", color='k'):
        # Plot each environment in a different subplot
        fig, axs = plt.subplots(1, len(self.environments), figsize=(5*len(self.environments), 5))
        for i, ax in enumerate(axs):
            ax.set_title(f"Environment {i+1}")
            ax.set_xlabel(xLabel)
            ax.set_ylabel(yLabel)
            ax.plot(episodes, rewards[i], color=color, label=f"{plotTitle}")
            ax.plot(episodes, avgReward[i], color='orange', label=f'Past 10 average {plotTitle}')
        plt.legend()
        plt.savefig(f'out/{plotTitle}.png')

        # plt.show()
    
    def plotResults(self, res, episodes, plotTitle="Results", xLabel="Episodes", yLabel="Number of Times", color='k'):
        # Plot each environment in a different subplot
        fig, axs = plt.subplots(1, len(self.environments), figsize=(5*len(self.environments), 5))
        for i, ax in enumerate(axs):
            ax.set_title(f"Environment {i+1}")
            ax.set_xlabel(xLabel)
            ax.set_ylabel(yLabel)
            ax.plot(episodes, res[i], color=color, label=f"{plotTitle}")
        plt.legend()
        plt.savefig(f'out/{plotTitle}.png')

        # plt.show()
    
    def plotTestResults(self, fails, actions, environments):
        X_axis = np.arange(len(environments)) 
  
        plt.bar(X_axis - 0.2, actions, 0.4, label = 'Actions') 
        plt.bar(X_axis + 0.2, fails, 0.4, label = 'Fails') 
        
        plt.xticks(X_axis, environments) 
        plt.xlabel("Environments") 
        plt.ylabel("Number of Instances") 
        plt.title("Number of Actions and Fails in each Environment") 
        plt.legend()
        plt.savefig(f'out/Testing_Results.png')

        # plt.show()
    
    def envStep(self, action, goal, controller, episode, fails):
        if action.objectOn is not None:
            controller.step(action=action.actionType, objectId=action.objectOn)
        else:
            if action.actionType in uc.MOVEMENT_ACTION_TYPES or action.actionType == "DropHandObject":
                controller.step(action=action.actionType)
            else:
                #TODO: If we reach here, assign a negative reward and use the previous state as nextState
                fails += 1
                pass
        nextState = State.State(controller.last_event.metadata,
                                controller.step("GetReachablePositions").metadata["actionReturn"])
        if not controller.last_event.metadata["lastActionSuccess"]:
            fails+=1
        reward = nextState.getReward(goal)
        terminated = self.goalCompleted(action, goal)
        if terminated:
            reward = 1
        return nextState, reward, terminated, (episode > 100 and not terminated), fails

    def optimizeModel(self):
        if len(self.memory) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, terminateds = zip(*self.memory.sample(BATCH_SIZE))
        state_batch = torch.tensor(np.array(states), device=self.device, dtype=torch.float)
        action_batch = torch.tensor(actions, device=self.device)
        reward_batch = torch.tensor(rewards, device=self.device)
        next_state_batch = torch.tensor(np.array(next_states), device=self.device, dtype=torch.float)
        terminated_batch = torch.tensor(terminateds, dtype=torch.int, device=self.device)
        
        state_action_values = self.policy_net.Q(state_batch, action_batch)
        expected_state_action_values = torch.zeros(BATCH_SIZE, device=self.device)
        ## state_action_values = Q(s,a, \theta) 
        ## expected_state_action_values = r + \gamma max_a Q(s',a, \theta_{tar})
        for i in range(BATCH_SIZE):
            if terminated_batch[i]:
                expected_state_action_values[i] = reward_batch[i]
            else:
                expected_state_action_values[i] = reward_batch[i] + GAMMA * torch.max(self.target_net(next_state_batch[i]))    
        loss = F.mse_loss(state_action_values, expected_state_action_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # Training the interactive agent in the ai2thor environments
    def train(self, controller):
        rewardsArray = []
        episodeArray = []
        failsArray = []
        actionsArray = []
        goalsCompletedArray = []
        for idx, environment in enumerate(tqdm(self.environments, desc=f"Training model for {len(self.environments)} environments")):
            goalsCompleted = 0
            goalsCompletedArray.append([])

            episode_returns = []
            average_returns = []
            ffArray = []
            aaArray = []

            # Reset the policy and target networks to initial params for new task
            self.policy_net.reset()
            self.target_net.reset()
            for epoch in tqdm(range(self.epochs), desc=f"Training the model for {self.epochs} epochs"):
                # Get environment specific goal
                goal = self.goalTasks[idx]

                fails = 0
                episode = 0

                # Reset the controller to the new environment
                controller.reset(scene=environment)
                controller.step(action="Initialize")
                state = State.State(controller.last_event.metadata,
                                    controller.step("GetReachablePositions").metadata["actionReturn"])

                terminated = 0
                truncated = 0
                current_episode_return = 0
                fails = 0
                while not(terminated or truncated):
                    state = State.State(controller.last_event.metadata, 
                                    controller.step("GetReachablePositions").metadata["actionReturn"])
                    eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * (epoch+episode) / EPS_DECAY)

                    act = 0
                    # Determine action
                    if random.random() > eps:
                        act = self.policy_net.maxQ(torch.tensor(state.getOneHotEncoding(), device=self.device, dtype=torch.float))
                    else:
                        act = uc.ALL_ACTIONS.index(random.sample(state.getPossibleActions(), 1)[0])

                    action = Action.Action(uc.ALL_ACTIONS[act], state.chooseFromReach(uc.ALL_ACTIONS[act], goal), False)
                    if self.verbose:
                        print(f"Choosing Action: {action}")
                    
                    nextState, reward, terminated, truncated, fails = self.envStep(action, goal, controller, episode, fails)
                    self.memory.push(state.getOneHotEncoding(), 
                                     act, reward, 
                                     nextState.getOneHotEncoding(), terminated)
                    current_episode_return += reward
                    
                    for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                        target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)
                    
                    if terminated or truncated:
                        goalsCompleted += 1 if terminated else 0
                        episode = 0
                        if self.verbose and epoch % 20 == 0:
                            print('Episode {},  score: {}'.format(epoch, current_episode_return))
                        
                        episode_returns.append(current_episode_return)
                        #### Store the average returns of 100 consecutive episodes
                        if epoch < 100:
                            average_returns.append(np.average(episode_returns))
                        else:
                            average_returns.append(np.average(episode_returns[epoch-100: epoch]))
                        ffArray.append(fails)
                        aaArray.append(episode-fails)
                    else:
                        state = nextState
                    
                    self.optimizeModel()
                    episode += 1
                
                # Clean up the AI2THOR environment
                controller.step(action="Done")
                
                if (self.verbose):
                    print(f"Goal: {goal}")
                    print(f"State to achieve goal: {state}")
                    print(f"The robot failed to do an action {fails} times.")

                if self.verbose and epoch % 50 == 0:
                    print(f"Number of Goals Completed: {goalsCompleted}\n")
                goalsCompletedArray[idx] += [goalsCompleted]
            
            # Stats on reward, fails, and actions for each environment
            episodeArray.append(episode_returns)
            rewardsArray.append(average_returns)
            failsArray.append(ffArray)
            actionsArray.append(aaArray)

            # Save this environment's trained policy and target networks
            self.saveModel("src/models/agent", idx)
            self.saveModel("src/models/agent", idx)

        # Plot the rewards, fails, actions, and goals completed across episodes and environments
        self.plotReward(episodeArray, rewardsArray, np.arange(self.epochs), plotTitle="Average_Rewards", xLabel="Episodes", yLabel="Average Rewards")
        self.plotResults(failsArray, np.arange(self.epochs), plotTitle="Fails", xLabel="Episodes", yLabel="Fails", color='b')
        self.plotResults(actionsArray, np.arange(self.epochs), plotTitle="Actions", xLabel="Episodes", yLabel="Actions", color='c')
        self.plotResults(goalsCompletedArray, np.arange(self.epochs), plotTitle="Goals_Completed", xLabel="Episodes", yLabel="Goals Completed")

    # Let's test the agent in the ai2thor environments
    def test(self, controller, model_path):
        fArray = []
        aArray = []
        for i, environment in enumerate(tqdm(self.environments, desc=f"Testing model for {len(self.environments)} environments")):
            self.policy_net.load_state_dict(torch.load(f"{model_path}_{i}_policy.pt"))
            self.target_net.load_state_dict(torch.load(f"{model_path}_{i}_target.pt"))
            fails = 0
            actions = 0
            controller.reset(scene=environment)

            # Initial state
            state = State.State(controller.last_event.metadata,
                                controller.step("GetReachablePositions").metadata["actionReturn"])
            newState = state
            goal = self.goalTasks[i]
            
            frames = []
            while(actions < 100):
                frames.append(controller.last_event.frame)
                # Determine action
                acts = self.policy_net(torch.tensor(state.getOneHotEncoding(), device=self.device, dtype=torch.float))
                pActIdx = [uc.ALL_ACTIONS.index(a) for a in state.getPossibleActions()]
                sortedActs, sortedIdx = torch.sort(acts, descending=True)
                sortedMovements = sortedIdx[[i for i in range(len(uc.ALL_ACTIONS)) if sortedIdx[i] < len(uc.MOVEMENT_ACTION_TYPES)]]
                sortedActions = sortedIdx[[i for i in range(len(uc.ALL_ACTIONS)) if sortedIdx[i] >= len(uc.MOVEMENT_ACTION_TYPES)]]

                act = sortedActions[0].item()
                idx = 1
                while(state.chooseFromReach(uc.ALL_ACTIONS[act], goal) != goal['objectId'] and state==newState):
                    if (idx >= (sortedActions).numel()):
                        act = sortedMovements[0].item()
                        break
                    act = sortedActions[idx].item()
                    idx+=1
                # act = self.policy_net.maxQ(torch.tensor(state.getOneHotEncoding(), device=self.device, dtype=torch.float))
                action = Action.Action(uc.ALL_ACTIONS[act], state.chooseFromReach(uc.ALL_ACTIONS[act], goal), False)
                
                # Execute action
                if action.objectOn is not None:
                    event = controller.step(action.actionType, objectId=action.objectOn) 
                else:
                    event = controller.step(action.actionType)

                newState = State.State(controller.last_event.metadata,
                                    controller.step("GetReachablePositions").metadata["actionReturn"])
                if newState == state:
                    fails += 1
                    newAct = Action.Action(action.getOppositeActionType(), action.objectOn, False)
                    if newAct.objectOn is not None:
                        event = controller.step(newAct.actionType, objectId=newAct.objectOn) 
                    else:
                        if newAct.actionType in uc.MOVEMENT_ACTION_TYPES or newAct.actionType == "DropHandObject":
                            controller.step(action=newAct.actionType)
                        else:
                            pass
                    action = newAct

                idx = 1
                while (not event.metadata["lastActionSuccess"]):
                    # Handle an unsuccessful action
                    # Try the next best action
                    fails += 1
                    idx+=1
                    if idx > (sortedActions).numel():
                        act = sortedMovements[0].item()
                        break
                    act = sortedActions[idx].item()
                    # while(state.chooseFromReach(uc.ALL_ACTIONS[act], goal) != goal['objectId'] and state==newState):
                    #     if (idx >= (sortedActions).numel()):
                    #         act = sortedMovements[0].item()
                    #         break
                    #     act = sortedActions[idx].item()
                    #     idx+=1
                    action = Action.Action(uc.ALL_ACTIONS[act], state.chooseFromReach(uc.ALL_ACTIONS[act], goal), False)
                
                    # Execute action
                    if action.objectOn is not None:
                        event = controller.step(action.actionType, objectId=action.objectOn) 
                    else:
                        if action.actionType in uc.MOVEMENT_ACTION_TYPES or action.actionType == "DropHandObject":
                            controller.step(action=action.actionType)
                        else:
                            pass

                    newState = State.State(controller.last_event.metadata,
                                        controller.step("GetReachablePositions").metadata["actionReturn"])
                
                actions += 1
                if (self.goalCompleted(action, goal)):
                    frames.append(controller.last_event.frame)
                    if self.verbose:
                        print(f"\nGoal for {environment} completed!")
                    break
                
                newState = State.State(controller.last_event.metadata,
                                    controller.step("GetReachablePositions").metadata["actionReturn"])
                state = newState

            fArray.append(fails)
            aArray.append(actions)
            
            controller.step(action="Done")
            if (self.verbose):
                print(f"The robot failed to do an action {fails} times.")
                print(f"The robot did {actions} actions.")
            imageio.mimsave(f'out/{environment}.mp4', frames, fps=1)

            self.policy_net.reset()
            self.target_net.reset()
        
        self.plotTestResults(fArray, aArray, self.environments)

