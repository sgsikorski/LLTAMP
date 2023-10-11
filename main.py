from ai2thor.controller import Controller
import time
import torch
import util
import lifelonglearning as ll

# Some global variables
environments = []
Buffer = []
BufferStream = []

# Learning threshold
threshold = 0.05

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

# Training the interactive agent in the ai2thor environments
def train():
    for environment in environments:
        # Init this environment's buffer memory to empty
        M = []
        # Reset the controller to the new environment
        controller.reset(scene=environment)

        # Initial state
        state = util.State(controller.last_event)

        # Run this environment actions for 100 seconds
        startTime = time.perf_counter()
        while(time.perf_counter - startTime < 100):
            # Determine action
            action = ll.InitialPolicy(state)
            transition = util.Transition(state, action)

            # Execute action
            event = controller.step(action = action["Type"])
            BufferStream.append(transition)

            # Select s_p and a_p from BufferStream
            if (ll.D(transition) < ll.threshold):
                newTransition = ll.A_correct(state, BufferStream)
                M.append(newTransition)

            newState = util.State(event.metadata)
            state = newState
        
        # Lifelong Learning Componenet
        theta_k = ll.A_cl()
        
# Let's test the agent in the ai2thor environments
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