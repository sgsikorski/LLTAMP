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