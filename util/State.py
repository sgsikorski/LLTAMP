import numpy as np
import json
from dataclasses import dataclass

@dataclass(frozen=True, order=True)
class State():
    def __init__(self, eventMetadata = None):
        self.envMD = eventMetadata
        self.agentX = eventMetadata['agent']['position']['x']
        self.agentY = eventMetadata['agent']['position']['y']
        self.agentZ = eventMetadata['agent']['position']['z']
        self.visibleObjects = [obj for obj in eventMetadata['objects'] if obj['visible']]
        self.reachableObjects = self.mapPosToObjs(eventMetadata["actionReturn"])
    
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
                and self.visibleObjects == other.visibleObjects
                and self.reachableObjects == other.reachableObjects)
    
    def getManhattanDistance(self):
        return np.sqrt(self.agentX**2 + self.agentY**2 + self.agentZ**2)
    
    def getObjManhattanDistance(self, obj):
        return np.sqrt(obj["position"]["x"]**2 + obj["position"]["y"]**2 + obj["position"]["z"]**2)

    def getObjDiff(self, otherState):
        inBoth = list(set(self.visibleObjects).intersection(set(otherState.visibleObjects)))
        numDiff = 10 * (len(self.visibleObjects) + len(otherState.visibleObjects) - len(inBoth)) 
        for obj in inBoth:
            numDiff += self.getObjManhattanDistance(obj) - otherState.getObjManhattanDistance(obj)
        return numDiff

    def mapPosToObjs(self, positions):
        reachableObjects = []
        for obj in self.visibleObjects:
            if (obj["position"] == positions):
                reachableObjects.append(obj)
        return reachableObjects