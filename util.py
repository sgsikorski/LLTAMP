"""
Utility file to help with global constants and some useful abstractions and methods
"""

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

ACTION_TYPES = [
    "MoveAhead",
    "MoveBack",
    "MoveLeft",
    "MoveRight",
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