"""
Utility file to help with global constants and some useful abstractions and methods
"""
def getPotentialActions(state):
    potentialActions = []
    if len(state.reachableObjects) != 0:
        for obj in state.reachableObjects:
            if (obj['pickupable']):
                potentialActions.append("PickupObject")
                potentialActions.append("PutObject")
            if (obj['moveable']):
                potentialActions.append("MoveHeldObject")
                potentialActions.append("PushObject")
    return potentialActions

def determineAction(status):
    match(status):
        case "open":
            return "OpenObject"
        case "closed":
            return "CloseObject"
    return

GRIDSIZE = 0.25

# Possible goal objects
TARGET_OBJECT_TYPES = [
    "AlarmClock,"
    "Apple,"
    "BaseballBat,"
    "BasketBall,"
    "Bowl,"
    "GarbageCan,"
    "HousePlant,"
    "Laptop,"
    "Mug,"
    "RemoteControl,"
    "SprayBottle,"
    "Television,"
    "Vase"
]

# All possible objects in scenes
BACKGROUND_OBJECT_TYPES = [
    "ArmChair",
    "Bed",
    "Book",
    "Bottle",
    "Box",
    "ButterKnife",
    "Candle",
    "CD",
    "CellPhone",
    "Chair",
    "CoffeeTable",
    "Cup",
    "DeskLamp",
    "Desk",
    "DiningTable",
    "Drawer",
    "Dresser",
    "FloorLamp",
    "Fork",
    "Newspaper",
    "Painting",
    "Pencil",
    "Pen",
    "PepperShaker",
    "Pillow",
    "Plate",
    "Pot",
    "SaltShaker",
    "Shelf",
    "SideTable",
    "Sofa",
    "Statue",
    "TeddyBear",
    "TennisRacket",
    "TVStand",
    "Watch"
]

MOVEMENT_ACTION_TYPES = [
    "MoveAhead",
    "MoveBack",
    "MoveLeft",
    "MoveRight"
]

ACTION_TYPES = [
    "PickupObject",
    "PutObject",
    "DropHandObject",
    "MoveHeldObject",
    "RotateHeldObject",
    "PushObject",
    "PullObject",
    "OpenObject",
    "CloseObject",
    "BreakObject",
    "CookObject",
    "SliceObject",
    "ToggleObjectOn",
    "ToggleObjectOff"
    "DirtyObject",
    "CleanObject",
    "UseUpObject",
    "FillObjectWithLiquid",
    "EmptyLiquidFromObject",
    "UseUpObject"
]