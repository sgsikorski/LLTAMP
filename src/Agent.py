import time
import torch
import torch.nn.functional as F
from util import State, Action, Transition
from util import utilConstants as uc
import lifelonglearning as ll
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
EPISODE_LENGTH = 300
EPS_DECAY = 50
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

        self.policy_net = ll.DQN(3+len(uc.OBJECT_TYPES)*(len(uc.OBJECT_PROPERTIES)+1), len(uc.ALL_ACTIONS)).to(self.device)
        self.target_net = ll.DQN(3+len(uc.OBJECT_TYPES)*(len(uc.OBJECT_PROPERTIES)+1), len(uc.ALL_ACTIONS)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LR, amsgrad=True)
    
    def saveModel(self, mp, envidx):
        torch.save(self.policy_net.state_dict(), f"{mp}_{envidx}_policy.pt")
        torch.save(self.target_net.state_dict(), f"{mp}_{envidx}_target.pt")

    def goalCompleted(self, action, goal) -> bool:
        return (action.actionType == uc.determineAction(goal['status']) and action.objectOn == goal['objectId'])
    
    def plotReward(self, rewards, episodes, plotTitle="Rewards", xLabel="Episodes", yLabel="Rewards", color='k'):
        # Plot each environment in a different subplot
        fig, axs = plt.subplots(1, len(self.environments), figsize=(5*len(self.environments), 5))
        for i, ax in enumerate(axs):
            ax.set_title(f"Environment {i+1}")
            ax.set_xlabel(xLabel)
            ax.set_ylabel(yLabel)
            ax.plot(episodes, rewards[i][0], color=color, label=f"{plotTitle}")
            ax.plot(episodes, rewards[i][1], color='orange', label=f'Past 10 average {plotTitle}')
        plt.legend()
        plt.savefig(f'out/{plotTitle}_4.png')

        # plt.show()
    
    def plotGoalCompletion(self, goals, episodes, plotTitle="Goal_Completion", xLabel="Episodes", yLabel="Goals Completed", color='k'):
        # Plot each environment in a different subplot
        fig, axs = plt.subplots(1, len(self.environments), figsize=(5*len(self.environments), 5))
        for i, ax in enumerate(axs):
            ax.set_title(f"Environment {i+1}")
            ax.set_xlabel(xLabel)
            ax.set_ylabel(yLabel)
            ax.plot(episodes, goals[i], color=color, label=f"{plotTitle}")
        plt.legend()
        plt.savefig(f'out/{plotTitle}_4.png')

        # plt.show()
    
    def plotTestResults(self, fails, actions, environments):
        plt.bar()
        X_axis = np.arange(len(environments)) 
  
        plt.bar(X_axis - 0.2, actions, 0.4, label = 'Actions') 
        plt.bar(X_axis + 0.2, fails, 0.4, label = 'Fails') 
        
        plt.xticks(X_axis, environments) 
        plt.xlabel("Environments") 
        plt.ylabel("Number of Instances") 
        plt.title("Number of Actions and Fails in each Environment") 
        plt.legend()
        plt.savefig(f'out/test.png')

        # plt.show()

    # Training the interactive agent in the ai2thor environments
    def train(self, controller):
        rewardsArray = []
        failsArray = []
        actionsArray = []
        goalsCompletedArray = []
        for idx, environment in enumerate(tqdm(self.environments, desc=f"Training model for {len(self.environments)} environments")):
            rArray = []
            fArray = []
            aArray = []
            avgR = []
            avgF = []
            avgA = []
            goalsCompleted = 0
            goalsCompletedArray.append([])

            # Reset the policy and target networks to initial params for new task
            self.policy_net.reset()
            self.target_net.reset()
            for epoch in tqdm(range(self.epochs), desc=f"Training the model for {self.epochs} epochs"):
                # Get environment specific goal
                goal = self.goalTasks[idx]

                fails = 0
                episode = 0
                rewards = []

                # Reset the controller to the new environment
                controller.reset(scene=environment)
                controller.step(action="Initialize")

                # Initial state
                state = State.State(controller.last_event.metadata, 
                                    controller.step("GetReachablePositions").metadata["actionReturn"])
                
                while(episode < EPISODE_LENGTH):
                    eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * (epoch+episode) / EPS_DECAY)

                    act = 0
                    # Determine action
                    if random.random() > eps:
                        act = self.policy_net.maxQ(torch.tensor(state.getOneHotEncoding(), device=self.device))
                        if (uc.ALL_ACTIONS[act] not in state.getPossibleActions()):
                            act = uc.ALL_ACTIONS.index(random.sample(state.getPossibleActions(), 1)[0])
                    else:
                        act = uc.ALL_ACTIONS.index(random.sample(state.getPossibleActions(), 1)[0])
                    
                    action = Action.Action(uc.ALL_ACTIONS[act], state.chooseFromReach(uc.ALL_ACTIONS[act], goal), False)
                    if self.verbose:
                        print(f"Choosing Action: {action}")
                    
                    # Execute action
                    if action.objectOn is not None:
                        event = controller.step(action.actionType, objectId=action.objectOn) 
                    else:
                        event = controller.step(action.actionType)

                    newState = State.State(controller.last_event.metadata,
                                        controller.step("GetReachablePositions").metadata["actionReturn"])
                    # Calculate the reward for the action from s->s'
                    reward = newState.getReward(goal)

                    if (not event.metadata["lastActionSuccess"]):
                        # Handle an unsuccessful action - Reward = -1
                        if self.verbose:
                            print(f"Action {action} failed.")
                        fails += 1
                        # continue
                    
                    for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                        target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)
                    
                    # We accomplished the goal, do additional reporting here and give a larger reward
                    if (self.goalCompleted(action, goal)):
                        if self.verbose:
                            print(f"\nGoal for {environment} completed!")

                        state_action_values = self.policy_net(torch.tensor(state.getOneHotEncoding(), device=self.device))[act]
                        expected_state_action_values = torch.tensor(10.0, dtype=torch.float64, device=self.device)
                        loss = F.mse_loss(state_action_values, expected_state_action_values)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        rewards.append(10.0)
                        goalsCompleted += 1
                        break
                    
                    # Update the policy network given the chosen (state, action, reward, newState) with MSE loss
                    state_action_values = self.policy_net(torch.tensor(state.getOneHotEncoding(), device=self.device))[act]
                    expected_state_action_values = reward + GAMMA * torch.max(self.target_net(torch.tensor(newState.getOneHotEncoding(), device=self.device)))
                    loss = F.mse_loss(state_action_values, expected_state_action_values)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                    rewards.append(reward)
                    # Create the new state and add it to the memory
                    newState = State.State(controller.last_event.metadata,
                                        controller.step("GetReachablePositions").metadata["actionReturn"])
                    state = newState
                    episode += 1

                    # rArray.append(reward)
                
                # Clean up the AI2THOR environment
                controller.step(action="Done")

                # Update this environment's information
                rArray.append(np.average(rewards))# / episode * EPISODE_LENGTH)
                fArray.append(fails)
                aArray.append(episode-fails)

                if epoch > 10:
                    avgR.append(np.average(rArray[-10:]))
                    avgF.append(np.average(fArray[-10:]))
                    avgA.append(np.average(aArray[-10:]))
                else:
                    avgR.append(np.average(rArray))
                    avgF.append(np.average(fArray))
                    avgA.append(np.average(aArray))
                
                # TODO: Lifelong Learning Componenet
                # self.model.update_theta(self.BufferStream, idx)

                if (self.verbose):
                    print(f"Goal: {goal}")
                    print(f"State to achieve goal: {state}")
                    print(f"The robot failed to do an action {fails} times.")

                if self.verbose and epoch % 50 == 0:
                    print(f"Number of Goals Completed: {goalsCompleted}\n")
                goalsCompletedArray[idx] += [goalsCompleted]
            
            rewardsArray.append((rArray, avgR))
            failsArray.append((fArray, avgF))
            actionsArray.append((aArray, avgA))

            if self.verbose:
                print(f"Epoch {epoch+1} completed!")

            # Save this environment's trained policy and target networks
            self.saveModel("src/models/agent", idx)
            self.saveModel("src/models/agent", idx)

        # Plot the rewards, fails, and actions
        self.plotReward(rewardsArray, np.arange(self.epochs), plotTitle="Average_Rewards", xLabel="Episodes", yLabel="Average Rewards")
        self.plotReward(failsArray, np.arange(self.epochs), plotTitle="Fails", xLabel="Episodes", yLabel="Fails", color='b')
        self.plotReward(actionsArray, np.arange(self.epochs), plotTitle="Actions", xLabel="Episodes", yLabel="Actions", color='c')
        self.plotGoalCompletion(goalsCompletedArray, np.arange(self.epochs), plotTitle="Goals_Completed", xLabel="Episodes", yLabel="Goals Completed")

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

            goal = self.goalTasks[i]
            
            frames = []
            while(True):
                frames.append(controller.last_event.frame)
                # Determine action
                acts = self.policy_net(torch.tensor(state.getOneHotEncoding(), device=self.device))
                pActIdx = [uc.ALL_ACTIONS.index(a) for a in state.getPossibleActions()]
                sortedActs, sortedIdx = torch.sort(acts[pActIdx], descending=True)
                act = sortedIdx[0].item()
                idx = 1
                if state.chooseFromReach(uc.ALL_ACTIONS[act], goal) is None and (act > 6 and act != 9 and act != 10):
                    act = sortedIdx[idx].item()
                    idx+=1
                action = Action.Action(uc.ALL_ACTIONS[act], state.chooseFromReach(uc.ALL_ACTIONS[act], goal), False)
                
                # Execute action
                if action.objectOn is not None:
                    event = controller.step(action.actionType, objectId=action.objectOn) 
                else:
                    event = controller.step(action.actionType)

                newState = State.State(controller.last_event.metadata,
                                    controller.step("GetReachablePositions").metadata["actionReturn"])
                idx = 0
                while (not event.metadata["lastActionSuccess"]):
                    # Handle an unsuccessful action
                    # Try the next best action
                    fails += 1
                    idx+=1
                    act = sortedIdx[idx].item()
                    action = Action.Action(uc.ALL_ACTIONS[act], state.chooseFromReach(uc.ALL_ACTIONS[act], goal), False)
                
                    # Execute action
                    if action.objectOn is not None:
                        event = controller.step(action.actionType, objectId=action.objectOn) 
                    else:
                        event = controller.step(action.actionType)

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
        
        self.plotTestResults(fArray, aArray, self.environments)

