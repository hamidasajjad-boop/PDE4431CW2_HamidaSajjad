# robot.py (corrected for exact wrist IK and origin at 0,0,0)

import numpy as np

class ArticulatedRRPR:
    def __init__(self,
                 a1=0.4,
                 a2=0.3,
                 a4=0.08,
                 base_height=0.0,      # FIXED: Origin is now exactly at z = 0
                 d3_min=0.0,
                 d3_max=0.8):

        self.a1 = a1
        self.a2 = a2
        self.a4 = a4
        self.base_height = base_height
        self.d3_min = d3_min
        self.d3_max = d3_max

        self.q1_min, self.q1_max = -np.pi, np.pi
        self.q2_min, self.q2_max = -np.pi, np.pi
        self.q4_min, self.q4_max = -np.pi, np.pi

    def forward_kinematics(self, q):
        theta1, theta2, d3, theta4 = q

        x = (self.a1*np.cos(theta1)
             + self.a2*np.cos(theta1+theta2)
             + self.a4*np.cos(theta1+theta2+theta4))
        y = (self.a1*np.sin(theta1)
             + self.a2*np.sin(theta1+theta2)
             + self.a4*np.sin(theta1+theta2+theta4))

        z = self.base_height + d3
        phi = theta1 + theta2 + theta4
        return np.array([x, y, z]), phi

    def inverse_kinematics(self, target, phi_desired=0.0, elbow_up=False):
        x, y, z = target

        # ---------------------------------------------------
        # 1) EXACT WRIST POSITION (critical fix)
        # ---------------------------------------------------
        Wx = x - self.a4*np.cos(phi_desired)
        Wy = y - self.a4*np.sin(phi_desired)

        r = np.hypot(Wx, Wy)

        # Law of cosines for theta2
        cos2 = (r**2 - self.a1**2 - self.a2**2) / (2*self.a1*self.a2)
        cos2 = np.clip(cos2, -1, 1)   # clamp

        sin2 = np.sqrt(1 - cos2**2)
        if elbow_up:
            sin2 = -sin2

        theta2 = np.arctan2(sin2, cos2)

        # Theta1
        k1 = self.a1 + self.a2*cos2
        k2 = self.a2*sin2

        theta1 = np.arctan2(Wy, Wx) - np.arctan2(k2, k1)

        # Prismatic joint
        d3 = z - self.base_height
        d3 = np.clip(d3, self.d3_min, self.d3_max)

        # Final wrist rotation
        theta4 = phi_desired - (theta1 + theta2)

        # Normalize
        theta1 = self._wrap(theta1)
        theta2 = self._wrap(theta2)
        theta4 = self._wrap(theta4)

        return np.array([theta1, theta2, d3, theta4])

    def _wrap(self, a):
        return (a + np.pi) % (2*np.pi) - np.pi

    def reachable(self, target):
        x, y, z = target
        r = np.hypot(x, y)
        max_r = self.a1 + self.a2 + self.a4
        return r <= max_r and self.d3_min <= (z - self.base_height) <= self.d3_max
