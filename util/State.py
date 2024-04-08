import numpy as np
import json
from dataclasses import dataclass, field
import torch
from util import utilConstants
import random

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
        inBoth = list(set(self.visObjName).intersection(set(otherState.visObjName)))
        numDiff = 10 * (len(self.visibleObjects) + len(otherState.visibleObjects) - len(inBoth))
        inBoth = [obj for obj in self.visibleObjects if obj["objectId"] in inBoth]
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

    def chooseFromReach(self, actionType):
        if (actionType == "MoveHeldObject"):
            return None
        if (actionType in utilConstants.MOVEMENT_ACTION_TYPES):
            return None
        toChoose = [obj['objectId'] for obj in self.reachableObjects if actionType in utilConstants.getPotentialActions(obj)]
        return random.choice(toChoose) if len(toChoose) > 0 else None

    def getPossibleActions(self):
        acts = []
        for o in self.reachableObjects:
            acts += utilConstants.getPotentialActions(o)
        return list(set(acts)) + utilConstants.MOVEMENT_ACTION_TYPES

    # This is really not a one-hot encoding as we have floats in the agent's position and the distance of the object
    def getOneHotEncoding(self):
        oneHot = [self.agentX, self.agentY, self.agentZ]
        oneHot += [0] * len(utilConstants.OBJECT_TYPES)
        for obj in self.visibleObjects:
            # If the object is not in the object types, we skip it
            if obj not in utilConstants.OBJECT_TYPES:
                continue
            if obj in self.reachableObjects:
                oneHot[utilConstants.OBJECT_TYPES.index(obj['objectType'])] = 1
            else:
                oneHot[utilConstants.OBJECT_TYPES.index(obj['objectType'])] = 1. / obj['distance']
        return oneHot

    def getReward(self, goal):
        if not self.envMD["lastActionSuccess"]:
            return -1
        if goal['objectId'] in self.reachObjName:
            return 1
        if goal['objectId'] in self.visObjName:
            return 0.5
        return 0
        