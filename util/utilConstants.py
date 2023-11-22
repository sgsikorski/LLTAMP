"""
Utility file to help with global constants and some useful abstractions and methods
"""
def getPotentialActions(obj):
    potentialActions = []
    if (obj['pickupable']):
        potentialActions.append("PickupObject")
        potentialActions.append("PutObject")
    if (obj['moveable']):
        potentialActions.append("MoveHeldObject")
        potentialActions.append("PushObject")
    if (obj['breakable']):
        potentialActions.append("BreakObject")
    if (obj['openable']):
        potentialActions.append("OpenObject")
        potentialActions.append("CloseObject")
    if (obj['canBeUsedUp']):
        potentialActions.append("UseUpObject")
    if (obj['canFillWithLiquid']):
        potentialActions.append("FillObjectWithLiquid")
    if (obj['cookable']):
        potentialActions.append("CookObject")
    if (obj['sliceable']):
        potentialActions.append("SliceObject")
    if (obj['dirtyable']):
        potentialActions.append("DirtyObject")
        potentialActions.append("CleanObject")
    if (obj['toggleable']):
        potentialActions.append("ToggleObjectOn")
        potentialActions.append("ToggleObjectOff")
    return potentialActions

def determineAction(status):
    match(status):
        case "Open":
            return "OpenObject"
        case "Closed":
            return "CloseObject"
        case "PickUp":
            return "PickupObject"
        case "Put":
            return "PutObject"
        case "Break":
            return "BreakObject"
        case "Cook":
            return "CookObject"
        case "Slice":
            return "SliceObject"
        case "Dirty":
            return "DirtyObject"
        case "Clean":
            return "CleanObject"
        case "UseUp":
            return "UseUpObject"
        case "Fill":
            return "FillObjectWithLiquid"
        case "MoveHeld":
            return "MoveHeldObject"
        case "Push":
            return "PushObject"
        case "On":
            return "ToggleObjectOn"
        case "Off":
            return "ToggleObjectOff"
        case _:
            return "MoveAhead"

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

ALL_OBJECT_TYPES = TARGET_OBJECT_TYPES + BACKGROUND_OBJECT_TYPES

MOVEMENT_ACTION_TYPES = [
    "MoveAhead",
    "MoveBack",
    "MoveLeft",
    "MoveRight",
    "RotateRight",
    "RotateLeft"
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
    "ToggleObjectOff",
    "DirtyObject",
    "CleanObject",
    "UseUpObject",
    "FillObjectWithLiquid",
    "EmptyLiquidFromObject",
    "UseUpObject"
]

ALL_ACTIONS = ["TASKDONE"] + MOVEMENT_ACTION_TYPES + ACTION_TYPES