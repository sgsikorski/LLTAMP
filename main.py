from ai2thor.controller import Controller
import time
import torch
import util

environments = []
Buffer = []
BufferStream = []

def InitialPolicy():
    action = {}
    action["Type"] = util.ACTION_TYPES[0]
    return action

def ThetaPolicy():

    return

def train():
    controller = Controller(
        agentMode = "locobot",
        visibilityDistance=1.5,
        scene=environments[0],
        gridSize=0.25,
        movementGaussianSigma = 0,
        rotateStepDegress=90,
        rotateGaussianSigma = 0,
        fieldOfView = 90
    )
    for environment in environments:
        # Init this environment's buffer memory to empty
        M = []
        # Reset the controller to the new environment
        controller.reset(scene=environment)

        # Initial state
        state = controller.last_event

        # Run this environment actions for 100 seconds
        startTime = time.perf_counter()
        while(time.perf_counter - startTime < 100):
            # Determine action
            action = InitialPolicy(state)
            transition = (state, action)

            # Execute action
            event = controller.step(action = action["Type"])
            BufferStream.append(transition)
            newState = event.metadata
            
            state = newState
        

def test():
    successRate = 0
    return successRate

def main():
    # Add all iThor scenes
    kitchens = [f"FloorPlan{i}" for i in range(1, 31)]
    living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 31)]
    bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 31)]
    bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 31)]

    environments = kitchens + living_rooms + bedrooms + bathrooms
    train()
    test()

    return

if __name__ == '__main__':
    main()