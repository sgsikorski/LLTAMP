import numpy as np
import json

class State():
    def __init__(self, eventMetadata = None):
        self.envMD = eventMetadata
        self.agentX = eventMetadata['agent']['position']['x']
        self.agentY = eventMetadata['agent']['position']['y']
        self.agentZ = eventMetadata['agent']['position']['z']
        self.visibleObjects = [obj for obj in eventMetadata['objects'] if obj['visible']]
    
    def __repr__(self) -> str:
        return f"""\u007b
    \"Visible Objects\": {json.dumps(self.visibleObjects, sort_keys=True, indent=4)},
    \"Agent\": \u007b 
        \"x\": {self.agentX},
        \"y\": {self.agentY},
        \"z\": {self.agentZ}
    \u007d
\u007d"""
    
    def __hash__(self):
        return hash((self.agentX, self.agentY, self.agentZ, [obj["objectId"] for obj in self.visibleObjects]))

    def __eq__(self, other):
        return (self.agentX == other.agentX 
                and self.agentY == other.agentY 
                and self.agentZ == other.agentZ 
                and self.visibleObjects == other.visibleObjects)
    
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

    def scanForGoal(self, controller, goalObject, rotateStepDegrees=90):
        for _ in range(0, 360, rotateStepDegrees):
            controller.step("RotateRight", degrees=rotateStepDegrees)
            if (self.visibleObjects is not None):
                for obj in self.visibleObjects:
                    if (obj["objectId"] == goalObject):
                        return obj
        return None