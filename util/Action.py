from dataclasses import dataclass

@dataclass(frozen=True, order=False)
class Action():
    actionType: str
    objectOn: str
    completeGoal: bool

    def __repr__(self) -> str:
        return f"""
Action: \u007b
    ActionType: {self.actionType}, 
    Object: {self.objectOn}, 
    Goal Complete: {self.completeGoal}
\u007d"""
    
    def getOppositeActionType(self):
        match(self.actionType):
            case "MoveAhead":
                return "MoveBack"
            case "MoveBack":
                return "MoveAhead"
            case "MoveLeft":
                return "MoveRight"
            case "MoveRight":
                return "MoveLeft"
            case "RotateRight":
                return "RotateLeft"
            case "RotateLeft":
                return "RotateRight"
            case "CloseObject":
                return "OpenObject"
            case "OpenObject":
                return "CloseObject"
            case "ToggleObjectOn":
                return "ToggleObjectOff"
            case "ToggleObjectOff":
                return "ToggleObjectOn"
            case _:
                return self.actionType