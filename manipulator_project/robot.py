import numpy as np


class ArticulatedRRRP:
    """
    4-DOF SCARA-style articulated manipulator:

        q1: base rotation      (about Z, at top of column)
        q2: shoulder rotation  (horizontal plane)
        q3: elbow rotation     (horizontal plane)
        q4: wrist vertical prismatic (short stroke)

    Geometry:
        Base frame at (0, 0, 0).
        Vertical column of height z_base supports 3R planar arm.
        q4 is a short vertical lift (10–20 cm), not a long rod.
    """

    def __init__(
        self,
        L1: float = 0.30,     # first link
        L2: float = 0.25,     # second link
        L3: float = 0.18,     # wrist link
        z_base: float = 0.25, # arm height
        z_min: float = -0.15,
        z_max: float = 0.15,  # ±15 cm wrist lift
    ):
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.z_base = z_base
        self.z_min = z_min
        self.z_max = z_max

    # --------------------------------------------------------------
    # Forward Kinematics
    # --------------------------------------------------------------
    def forward_kinematics(self, q):
        q1, q2, q3, q4 = [float(v) for v in q]

        # base on floor
        p0 = np.array([0.0, 0.0, 0.0])

        # top of the column
        p1 = np.array([0.0, 0.0, self.z_base])

        # planar angles
        t1 = q1
        t2 = q1 + q2
        t3 = q1 + q2 + q3

        # shoulder
        p2 = p1 + np.array([
            self.L1 * np.cos(t1),
            self.L1 * np.sin(t1),
            0.0
        ])

        # elbow
        p3 = p2 + np.array([
            self.L2 * np.cos(t2),
            self.L2 * np.sin(t2),
            0.0
        ])

        # wrist
        wrist = p3 + np.array([
            self.L3 * np.cos(t3),
            self.L3 * np.sin(t3),
            0.0
        ])

        # small prismatic lift only
        q4_clamped = np.clip(q4, self.z_min, self.z_max)

        p4 = wrist.copy()
        p4[2] = self.z_base + q4_clamped

        return np.vstack([p0, p1, p2, p3, p4])

    # --------------------------------------------------------------
    # Inverse Kinematics
    # --------------------------------------------------------------
    def inverse_kinematics(self, target):
        x, y, z = target

        # wrist lift
        q4 = np.clip(z - self.z_base, self.z_min, self.z_max)

        # projection onto arm plane
        phi = np.arctan2(y, x)

        # wrist XY position
        xw = x - self.L3 * np.cos(phi)
        yw = y - self.L3 * np.sin(phi)

        L1, L2 = self.L1, self.L2

        r2 = xw**2 + yw**2
        r = np.sqrt(r2)
        max_r = L1 + L2 - 1e-6

        # clamp
        if r > max_r:
            xw *= max_r / r
            yw *= max_r / r
            r = max_r
            r2 = r * r

        # COSINE LAW
        c2 = (r2 - L1**2 - L2**2) / (2 * L1 * L2)
        c2 = np.clip(c2, -1, 1)
        s2 = np.sqrt(1 - c2**2)
        q2 = np.arctan2(s2, c2)

        k1 = L1 + L2 * c2
        k2 = L2 * s2

        q1 = np.arctan2(yw, xw) - np.arctan2(k2, k1)

        q3 = phi - q1 - q2

        return np.array([q1, q2, q3, q4])
