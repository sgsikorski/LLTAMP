"""
Utility file to help with global constants and some useful abstractions and methods
"""
def getPotentialActions(obj):
    potentialActions = []
    if (obj['pickupable']):
        potentialActions.append("PickupObject")
        potentialActions.append("DropHandObject")
        #potentialActions.append("PutObject")
        #if obj['moveable']:
        #    potentialActions.append("MoveHeldObject")
    #if (obj['moveable']):
    #    potentialActions.append("PushObject")
    #if (obj['breakable']):
    #    potentialActions.append("BreakObject")
    if (obj['openable']):
        potentialActions.append("OpenObject")
        potentialActions.append("CloseObject")
    #if (obj['canBeUsedUp']):
    #    potentialActions.append("UseUpObject")
    #if (obj['canFillWithLiquid']):
    #    potentialActions.append("FillObjectWithLiquid")
    #if (obj['cookable']):
    #    potentialActions.append("CookObject")
    #if (obj['sliceable']):
    #    potentialActions.append("SliceObject")
    #if (obj['dirtyable']):
    #    potentialActions.append("DirtyObject")
    #    potentialActions.append("CleanObject")
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
        #case "Break":
        #    return "BreakObject"
        #case "Cook":
        #    return "CookObject"
        #case "Slice":
        #    return "SliceObject"
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

# Objects that show up in all kitchen scenes. We can do one-hot encoding for these
OBJECT_TYPES = [
    "Apple",
    #"AppleSliced",
    "Bowl",
    "Bread",
    #"BreadSliced",
    "ButterKnife",
    "Cabinet",
    "CoffeeMachine",
    #"CounterTop",
    "Cup",
    #"DishSponge",
    "Egg",
    #"EggCracked",
    "Faucet",
    #"Floor",
    "Fork",
    "Fridge",
    "GarbageCan",
    "Knife",
    #"Lettuce",
    #"LettuceSliced",
    "LightSwitch",
    #"Microwave",
    #"Mug",
    #"Pan",
    #"PepperShaker",
    #"Plate",
    #"Pot",
    "Potato",
    #"PotatoSliced",
    "SaltShaker",
    "Sink",
    #"SinkBasin",
    #"SoapBottle",
    #"Spatula",
    #"Spoon",
    #"StoveBurner",
    #"StoveKnob",
    "Toaster",
    #"Tomato",
    #"TomatoSliced",
]

MOVEMENT_ACTION_TYPES = [
    "MoveAhead",
    "MoveBack",
    "MoveRight",
    "MoveLeft",
    "RotateRight",
    "RotateLeft"
]

ACTION_TYPES = [
    "PickupObject",
    #"PutObject",
    "DropHandObject",
    #"MoveHeldObject",
    #"RotateHeldObject",
    #"PushObject",
    #"PullObject",
    "OpenObject",
    "CloseObject",
    #"BreakObject",
    #"CookObject",
    #"SliceObject",
    "ToggleObjectOn",
    "ToggleObjectOff",
    #"DirtyObject",
    #"CleanObject",
    #"UseUpObject",
    #"FillObjectWithLiquid",
    #"EmptyLiquidFromObject",
    #"UseUpObject"
]

ALL_ACTIONS = MOVEMENT_ACTION_TYPES + ACTION_TYPES

OBJECT_PROPERTIES = [
    "isBroken",
    "isColdSource",
    "isCooked",
    "isDirty",
    "isFilledWithLiquid",
    "isHeatSource",
    "isInteractable",
    "isMoving",
    "isOpen",
    "isPickedUp",
    "isSliced",
    "isToggled",
    "isUsedUp"
]