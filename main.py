import cv2
import numpy as np
import mediapipe as mp
import time
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class Joint:
    name: str
    coordinates: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Angle in degrees


class HandSkeleton:
    def __init__(self, label: str):
        self.label = label  # 'Left' or 'Right'
        self.joints: Dict[str, Joint] = {}
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
        if name in self.joints:
            self.joints[name].coordinates = coords
            if rotation:
                self.joints[name].rotation = rotation

    def get_joint(self, name: str) -> Joint:
        return self.joints.get(name)

    def __str__(self):
        return f"\n{self.label} Hand:\n" + "\n".join(
            f"{name}: {joint.coordinates}, Rot: {joint.rotation}" for name, joint in self.joints.items()
        )

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ab = a - b
    cb = c - b

    ab_norm = ab / (np.linalg.norm(ab) + 1e-6)
    cb_norm = cb / (np.linalg.norm(cb) + 1e-6)

    dot_product = np.dot(ab_norm, cb_norm)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))

    # Use z-axis as reference for sign (assuming 2D projection on screen)
    cross = np.cross(ab_norm, cb_norm)
    sign = np.sign(cross[2])  # sign of z-component of cross product

    angle_deg = np.degrees(angle_rad) * sign
    return angle_deg


FINGER_JOINTS = {
    "THUMB": [("THUMB_CMC", "THUMB_MCP", "THUMB_IP"), ("THUMB_MCP", "THUMB_IP", "THUMB_TIP")],
    "INDEX": [("INDEX_MCP", "INDEX_PIP", "INDEX_DIP"), ("INDEX_PIP", "INDEX_DIP", "INDEX_TIP")],
    "MIDDLE": [("MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP"), ("MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP")],
    "RING": [("RING_MCP", "RING_PIP", "RING_DIP"), ("RING_PIP", "RING_DIP", "RING_TIP")],
    "PINKY": [("PINKY_MCP", "PINKY_PIP", "PINKY_DIP"), ("PINKY_PIP", "PINKY_DIP", "PINKY_TIP")]
}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# FPS tracking
prev_time = 0

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, hand_label in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = hand_label.classification[0].label.upper()  # 'Left' or 'Right'
                skeleton = HandSkeleton(label)

                for idx, landmark in enumerate(hand_landmarks.landmark):
                    name = skeleton.joint_names[idx]
                    x, y, z = landmark.x, landmark.y, landmark.z
                    skeleton.update_joint(name, (x, y, z))

                # Calculate angles
                for finger in FINGER_JOINTS:
                    for a, b, c in FINGER_JOINTS[finger]:
                        angle = calculate_angle(
                            skeleton.joints[a].coordinates,
                            skeleton.joints[b].coordinates,
                            skeleton.joints[c].coordinates
                        )
                        skeleton.update_joint(b, skeleton.joints[b].coordinates, (angle, 0.0, 0.0))

                        # Draw angle on frame
                        cx = int(skeleton.joints[b].coordinates[0] * frame.shape[1])
                        cy = int(skeleton.joints[b].coordinates[1] * frame.shape[0])
                        cv2.putText(frame, f'{int(angle)}', (cx, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # === Calculate and show FPS ===
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Both Hands Skeleton with Rotations and FPS", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

cap.release()
cv2.destroyAllWindows()
