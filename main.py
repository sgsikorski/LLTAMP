from ai2thor.controller import Controller
from util import utilConstants
import lifelonglearning as ll
import json
import argparse
from Agent import Agent

def main():
    parser = argparse.ArgumentParser(description="Training and testing an interactive robot for task completion in iTHOR")
    parser.add_argument("-tr", "--train", action="store_true", help="Train the model")
    parser.add_argument("-te", "--test", action="store_true", help="Test the model")    
    parser.add_argument("-mp", "--model_path", default='models/trained_agent.pt', dest='model_path', type=str, help="Path to the model")
    parser.add_argument("-gp", "--goal_path", dest='goal_path', default="goalTasks.json", type=str, help="Path to the goal tasks")
    parser.add_argument("-envs", "--environments", type=str, help="Environments to train and test the model on")
    parser.add_argument("-enum", "--enum", type=int, choices=range(1, 31), default=31, help="Number of environments to train and test the model on")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("-t", "--timeout", type=int, default=100, help="Timeout for training and testing")
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
    
    agent = Agent(environments, goalTasks, model, args.verbose, args.timeout)
    if args.train:
        agent.train(controller)
        model.saveModel(args.model_path)
    if args.test:
        model.loadModel(args.model_path)
        agent.test(controller)

    print("Done!")

if __name__ == '__main__':
    main()