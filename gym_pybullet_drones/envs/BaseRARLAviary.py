import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

class BaseRARLAviary(BaseRLAviary):
    """Base single and multi-agent environment class for reach-avoid reinforcement learning."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # define target set 
        self.TARGET_CENTER = np.array([2.0, 2.0, 1.0])
        self.TARGET_RADIUS = 0.5

        # define the unsafe set
        # where you will place obstacles in the enviorment (using URDFs)
        self.UNSAFE_REGIONS = [
            np.array([1.0, 1.0, 0.5]),
            np.array([-1.0, 0.0, 0.5])
        ]
        self.SAFETY_MARGIN = 0.5

    def target_function(self, drone_id=0):
        """
        Returns True if the drone is within the target set, False otherwise.
        """

        pos = self.getDroneStateVector(drone_id=drone_id)[0:3]
        return np.linalg.norm(pos - self.TARGET_CENTER) < self.TARGET_RADIUS
    
    def margin_function(self, drone_id=0):
        """ Returns minimum distance to the unsafe set. """
        pos = self.getDroneStateVector(drone_id=drone_id)[0:3]
        distance = [np.linalg.norm(pos - unsafe_region) for unsafe_region in self.UNSAFE_REGIONS]
        return min(distance)
    
    def _computeReward(self):
        """reward shaped for reach-avoid problem"""

        reward = 0.0
        for i in range(self.NUM_DRONES):
            if self.target_function(i):
                reward += 100.0

            margin = self.margin_function(i)

            if margin > self.SAFETY_MARGIN:
                reward -= (self.SAFETY_MARGIN - margin) * 50.0

            return reward 


    def _addObstacles(self):
        """Add obstacles to the environment (reusing BaseRLAviary RGB obstacles)"""
        super().addObstacles()
        for pos in self.UNSAFE_REGIONS:
            p.loadURDF("cube_small.urdf, pos, p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=self.CLIENT)")


    def _computeTerminated(self):
        """ Episode ends when goal is reached or margin is violated"""
        for i in range(self.NUM_DRONES):
            if self.target_function(i):
                return True
            if self.margin_function(i) < 0.05:
                return True
        return False
    
    def _computeTruncated(self):
        """Never truncate unless defined externally"""
        return False
    
    def _computeInfo(self):
        return {}