from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import time

@dataclass
class Joint:
    name: str
    coordinates: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)

class HandSkeleton:
    def __init__(self, label: str, update_interval=0.05, threshold=0.01, smoothing=0.1):
        self.label = label
        self.joints: Dict[str, Joint] = {}
        self.last_update_time = time.time()
        self.update_interval = update_interval  # in seconds
        self.threshold = threshold
        self.smoothing = smoothing

        self.joint_names = [
            "WRIST",
            "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
            "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
            "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
            "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
            "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
        ]

        for name in self.joint_names:
            self.joints[name] = Joint(name=name)

    def update_joint(self, name: str, coords: Tuple[float, float, float], rotation: Tuple[float, float, float] = None):
        current_time = time.time()
        if name not in self.joints:
            return

        joint = self.joints[name]
        old_coords = np.array(joint.coordinates)
        new_coords = np.array(coords)
        delta = np.linalg.norm(new_coords - old_coords)

        if delta < self.threshold:
            return  # Not enough movement to bother updating

        if current_time - self.last_update_time >= self.update_interval:
            smoothed_coords = old_coords + self.smoothing * (new_coords - old_coords)
            joint.coordinates = tuple(smoothed_coords)
            if rotation:
                joint.rotation = rotation
            self.last_update_time = current_time

@staticmethod
def calculate_rotation_axes(a, b, c):
    ab = np.array(a) - np.array(b)
    cb = np.array(c) - np.array(b)
    # Cross product gives normal vector
    normal = np.cross(ab, cb)
    
    # Calculate Euler-like interpretation or directional component
    rot_x = np.arctan2(normal[1], normal[2])
    rot_y = np.arctan2(normal[0], normal[2])
    rot_z = np.arctan2(normal[0], normal[1])
    
    return np.degrees([rot_x, rot_y, rot_z])