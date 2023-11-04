from dataclasses import dataclass

@dataclass(frozen=True, order=True)
class Action():
    actionType: str
    objectOn: str
    completeGoal: bool