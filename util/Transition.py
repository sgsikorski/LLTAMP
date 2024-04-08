from dataclasses import dataclass, field
from util.State import State
from util.Action import Action

@dataclass(frozen = True, order=False)
class Transition():
    state: State
    action: Action
    reward: float
    nextState: State

    def updateTransition(self, state = None, action = None, reward = None, nextState = None):
        if state is not None:
            self.state = state
        if action is not None:
            self.action = action
        if reward is not None:
            self.reward = reward
        if nextState is not None:
            self.nextState = nextState
    
    def __repr__(self) -> str:
        return f"{self.state}{self.action}"