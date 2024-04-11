import sys 
import os
import math
import torch

from ai2thor.controller import Controller
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)))
from util import utilConstants as uc
import lifelonglearning as ll
import json
import argparse
from Agent import Agent

def main():
    parser = argparse.ArgumentParser(description="Training and testing an interactive robot for task completion in iTHOR")
    parser.add_argument("-tr", "--train", action="store_true", help="Train the model")
    parser.add_argument("-te", "--test", action="store_true", help="Test the model")    
    parser.add_argument("-mp", "--model_path", default='src/models/trained_agent', dest='model_path', type=str, help="Path to the model")
    parser.add_argument("-gp", "--goal_path", dest='goal_path', default="goalTasks.json", type=str, help="Path to the goal tasks")
    parser.add_argument("-envs", "--environments", type=str, help="Environments to train and test the model on")
    parser.add_argument("-enum", "--enum", type=int, choices=range(1, 31), default=31, help="Number of environments to train and test the model on")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("-t", "--timeout", type=int, default=100, help="Timeout for training and testing")
    parser.add_argument("-e", "--epochs", type=int, default=5, help="Number of epochs to train the model")
    args = parser.parse_args()

    controller = Controller(
        agentMode = "default",
        visibilityDistance=1.5,
        scene="FloorPlan5",
        gridSize=uc.GRIDSIZE,
        movementGaussianSigma = 0,
        rotateStepDegress=90,
        rotateGaussianSigma = 0,
        fieldOfView = 90,
        width=960,
        height=1080
    )

    # input_dim: agent position + one-hot encoding of interactable objects
    model = ll.Model(input_dim=3+len(uc.OBJECT_TYPES), n_tasks=args.enum)

    # Add which environments we want to conduct over
    environments = []
    kitchens = [f"FloorPlan{i}" for i in range(1, args.enum+1)]
    living_rooms = [f"FloorPlan{200 + i}" for i in range(1, args.enum+1)]
    bedrooms = [f"FloorPlan{300 + i}" for i in range(1, args.enum+1)]
    bathrooms = [f"FloorPlan{400 + i}" for i in range(1, args.enum+1)]
    
    envsToUse = args.environments if args.environments is not None else "klbeba"
    if "k" in envsToUse: environments += kitchens
    if "l" in envsToUse: environments += living_rooms
    if "be" in envsToUse: environments += bedrooms
    if "ba" in envsToUse: environments += bathrooms

    # with open("objects.json", "w+") as f:
    #     f.write("{\n")
    #     for env in environments:
    #         controller.reset(scene=env)
    #         f.write("\"" + env + "\":[\n")
    #         for obj in controller.last_event.metadata["objects"]:
    #             f.write("\"" + obj['objectId'] + "\"")
    #             f.write(",\n")
    #         f.write("],\n")
    #     f.write("}")

    try:
        with open(args.goal_path) as f:
            goalTasks = json.load(f)
    except FileNotFoundError as e:
        print(f"{e}. Please define a goal tasks .json file accordingly. Now terminating")
        return
    
    agent = Agent(args, environments, goalTasks, model)
    if args.train:
        agent.train(controller)
        torch.save(agent.policy_net.state_dict(), f"{args.model_path}_policy.pth")
        torch.save(agent.target_net.state_dict(), f"{args.model_path}_target.pth")
    if args.test:
        # agent.model.loadModel(args.model_path)
        agent.policy_net.load_state_dict(torch.load(f"{args.model_path}_policy.pth"))
        agent.target_net.load_state_dict(torch.load(f"{args.model_path}_target.pth"))
        agent.test(controller)

    print("Done!")

if __name__ == '__main__':
    main()