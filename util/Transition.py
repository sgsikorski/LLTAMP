from dataclasses import dataclass, field
from util.State import State
from util.Action import Action

@dataclass(frozen = True, order=False)
class Transition():
    state: State
    action: Action

    def updateTransition(self, state = None, action = None):
        if state is not None:
            self.state = state
        if action is not None:
            self.action = action
    
    def __repr__(self) -> str:
        return f"{self.state}{self.action}"