from dataclasses import dataclass, field
from util.State import State
from util.Action import Action

@dataclass(frozen = True, order=True)
class Transition():
    state: State
    action: Action

    def updateTransition(self, state = None, action = None):
        if state is not None:
            self.state = state
        if action is not None:
            self.action = action
    # Maybe add some more useful methods