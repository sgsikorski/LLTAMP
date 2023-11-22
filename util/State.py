import numpy as np
import json
from dataclasses import dataclass, field
import torch

@dataclass(init = False, repr=False, frozen=False, order=False)
class State():
    envMD: dict
    agentX: float
    agentY: float
    agentZ: float
    visibleObjects: list = field(default_factory=list)
    visObjName: list= field(default_factory=list)
    reachableObjects: list = field(default_factory=list)
    reachObjName: list = field(default_factory=list)

    def __init__(self, envMD, reachMD):
        self.envMD = envMD
        self.agentX = envMD['agent']['position']['x']
        self.agentY = envMD['agent']['position']['y']
        self.agentZ = envMD['agent']['position']['z']
        self.visibleObjects = [obj for obj in envMD['objects'] if obj['visible']]
        self.visObjName = [obj['objectId'] for obj in self.visibleObjects]
        self.reachableObjects = self.mapPosToObjs(reachMD)
        self.reachObjName = [obj['objectId'] for obj in self.reachableObjects]
    
    def __repr__(self) -> str:
        return f"""
State: \u007b
    \"Visible Objects\": {self.visObjName},
    \"Reachable Objects\": {self.reachObjName},
    \"Agent\": \u007b 
        \"x\": {self.agentX},
        \"y\": {self.agentY},
        \"z\": {self.agentZ}
    \u007d
\u007d"""
    
    def __hash__(self):
        return hash((self.agentX, self.agentY, self.agentZ, tuple([obj["objectId"] for obj in self.visibleObjects])))

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
            if (obj["distance"] <= 1.0):
                reachableObjects.append(obj)
                break
        return reachableObjects