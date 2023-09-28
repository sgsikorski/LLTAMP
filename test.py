from ai2thor.controller import Controller

controller = Controller(
agentMode = "locobot",
visibilityDistance=1.5,
scene="FloorPlan_Train1_3",
gridSize=0.25
)

controller.step("MoveBack")
controller.step("MoveAhead")