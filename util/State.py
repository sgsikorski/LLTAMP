import numpy as np
import json

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
    
    def getManhattanDistance(self):
        return np.sqrt(self.envMD['agent']['position']['x']**2 
                       + self.envMD['agent']['position']['y']**2 
                       + self.envMD['agent']['position']['z']**2)

    def getNumOfDifferentVisibleObjects(self, otherState):
        count = 0 
        for (obj1, obj2) in (self.visibleObjects, otherState.visibleObjects):
            if obj1["ObjectId"] is not obj2["ObjectId"]:
                count += 1
        return count