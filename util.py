"""
Utility file to help with global constants and some useful abstractions and methods
"""
import json

class Transition():
    def __init__(self, state=None, action=None):
        if state is None or action is None:
            raise TypeError("State and action must be defined types for a transition")
        self.state = state
        self.action = action

    def updateTransition(self, state = None, action = None):
        if state is not None:
            self.state = state
        if action is not None:
            self.action = action
    # Maybe add some more useful methods

class State():
    def __init__(self, eventMetadata = None):
        self.envMD = eventMetadata
        self.visibleObjects = [obj for obj in eventMetadata['objects'] if obj['visible']]
    
    def __repr__(self) -> str:
        return f"""\u007b
    \"Visible Objects\": {json.dumps(self.visibleObjects, sort_keys=True, indent=4)},
    \"Agent\": \u007b 
        \"x\": {self.envMD['agent']['position']['x']},
        \"y\": {self.envMD['agent']['position']['y']},
        \"z\": {self.envMD['agent']['position']['z']}
    \u007d
\u007d"""

class Action():
    def __init__(self, actionType = None, objectOn = None):
        self.actionType = actionType
        self.objectOn = objectOn

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
    "ThrowObject",
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