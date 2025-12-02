import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from robot import ArticulatedRRRP


class ManipulatorSimulation:
    """
    4-DOF articulated manipulator performing pick & place
    for three floor objects onto three shelf levels.
    """

    def __init__(self):

        # ------------------------------
        # Robot model with corrected Z range
        # ------------------------------
        self.robot = ArticulatedRRRP(
            z_min=-0.23,   # allows reaching floor z ≈ 0.02
            z_max=0.40     # allows reaching shelf 3 z ≈ 0.65
        )

        # ------------------------------
        # Correct Home Configuration
        # q4 is extension, not absolute Z
        # world Z = 0.25 + 0.10 = 0.35
        # ------------------------------
        self.q_home = np.array([0.0, 0.0, 0.0, 0.10])
        self.q_current = self.q_home.copy()

        # ------------------------------
        # Floor objects
        # ------------------------------
        self.floor_positions = np.array(
            [
                [0.35, -0.15, 0.02],  
                [0.40,  0.00, 0.02],  
                [0.35,  0.15, 0.02],  
            ]
        )

        # ------------------------------
        # Shelf geometry
        # ------------------------------
        self.shelf_x = 0.40
        self.shelf_y = 0.52
        self.shelf_levels = np.array([0.25, 0.45, 0.65])

        self.shelf_positions = np.array(
            [
                [self.shelf_x, self.shelf_y, self.shelf_levels[0]],
                [self.shelf_x, self.shelf_y, self.shelf_levels[1]],
                [self.shelf_x, self.shelf_y, self.shelf_levels[2]],
            ]
        )

        # Track placed objects
        self.object_done = [False, False, False]

        # ------------------------------
        # Matplotlib figure
        # ------------------------------
        self.fig = plt.figure(figsize=(10, 7))
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.ax.set_title("4-DOF Articulated Manipulator – Pick & Place", fontsize=14)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")

        self.ax.set_xlim(-0.7, 0.7)
        self.ax.set_ylim(-0.5, 0.7)
        self.ax.set_zlim(0.0, 0.9)
        self.ax.view_init(elev=25, azim=40)

        # Draw environment
        self._draw_floor()
        self._draw_shelf()
        self._create_objects()
        self._draw_home_reference()

        # Draw robot at home
        joints = self.robot.forward_kinematics(self.q_current)
        (self.robot_line,) = self.ax.plot(
            joints[:, 0], joints[:, 1], joints[:, 2],
            "-o", color="red", linewidth=2, markersize=6
        )

        # Buttons (fixed)
        self._create_buttons()


    # --------------------------------------------------------------
    #  UI: Buttons
    # --------------------------------------------------------------
    def _create_buttons(self):
        names = ["Floor 1", "Floor 2", "Floor 3", "Reset"]
        start_y = 0.75
        dy = 0.08

        self.buttons = {}

        for i, name in enumerate(names):
            ax_btn = self.fig.add_axes([0.03, start_y - i * dy, 0.12, 0.06])
            btn = Button(ax_btn, name)

            if name == "Reset":
                btn.on_clicked(lambda event, self=self: self.go_home())

            else:
                idx = int(name.split()[-1]) - 1
                btn.on_clicked(lambda event, k=idx, self=self: self.pick_and_place(k))

            self.buttons[name] = btn


    # --------------------------------------------------------------
    # Environment
    # --------------------------------------------------------------
    def _draw_floor(self):
        x = np.linspace(-0.7, 0.7, 2)
        y = np.linspace(-0.5, 0.5, 2)
        xx, yy = np.meshgrid(x, y)
        zz = np.zeros_like(xx)
        self.ax.plot_surface(xx, yy, zz, color="lightgray", alpha=0.2)

    def _draw_shelf(self):
        width = 0.35
        depth = 0.18

        x_left = self.shelf_x - width / 2
        x_right = self.shelf_x + width / 2
        y_front = self.shelf_y
        y_back = self.shelf_y + depth

        z_top = self.shelf_levels[-1] + 0.05

        # Posts
        for x_post in [x_left, x_right]:
            self.ax.plot([x_post, x_post], [y_front, y_front], [0, z_top], color="dimgray", linewidth=3)
            self.ax.plot([x_post, x_post], [y_back, y_back], [0, z_top], color="dimgray", linewidth=3)

        # Shelf plates
        colors = ["lightblue", "peachpuff", "lightgreen"]
        for z, col in zip(self.shelf_levels, colors):
            xx = np.array([[x_left, x_right], [x_left, x_right]])
            yy = np.array([[y_front, y_front], [y_back, y_back]])
            zz = np.array([[z, z], [z, z]])
            self.ax.plot_surface(xx, yy, zz, color=col, alpha=0.7, edgecolor="k", linewidth=0.4)

        for i, z in enumerate(self.shelf_levels, start=1):
            self.ax.text(x_right + 0.02, y_back, z + 0.02, f"Shelf {i}", fontsize=8)

    def _create_objects(self):
        self.objects_pos = self.floor_positions.copy()
        self.objects_scatters = []

        colors = ["tab:blue", "tab:orange", "tab:green"]

        for i, p in enumerate(self.objects_pos):
            sc = self.ax.scatter(p[0], p[1], p[2], s=50, marker="s", color=colors[i])
            self.objects_scatters.append(sc)
            self.ax.text(p[0], p[1], p[2] + 0.03, f"O{i+1}", fontsize=8)

    def _draw_home_reference(self):
        ee = self.robot.forward_kinematics(self.q_home)[-1]
        self.ax.scatter(ee[0], ee[1], ee[2], s=60, color="magenta")
        self.ax.text(ee[0], ee[1], ee[2] + 0.03, "Home ref", color="magenta", fontsize=8)


    # --------------------------------------------------------------
    # Animation helper
    # --------------------------------------------------------------
    def _animate_segment(self, q_target, steps=60, carry_index=None):

        q_start = self.q_current.copy()

        for i in range(steps + 1):
            alpha = i / steps
            q = (1 - alpha) * q_start + alpha * q_target
            joints = self.robot.forward_kinematics(q)

            self.robot_line.set_data(joints[:, 0], joints[:, 1])
            self.robot_line.set_3d_properties(joints[:, 2])

            if carry_index is not None:
                ee = joints[-1]
                self.objects_pos[carry_index] = ee
                sc = self.objects_scatters[carry_index]
                sc._offsets3d = ([ee[0]], [ee[1]], [ee[2]])

            plt.pause(0.01)

        self.q_current = q_target.copy()


    # --------------------------------------------------------------
    # Pick-and-place with corrected detach logic
    # --------------------------------------------------------------
    def pick_and_place(self, obj_idx):

        if self.object_done[obj_idx]:
            return

        pick_pos = self.floor_positions[obj_idx]
        place_pos = self.shelf_positions[obj_idx]

        safe_height = 0.60

        # IK targets
        pre_pick = pick_pos.copy();  pre_pick[2] = safe_height
        pre_place = place_pos.copy(); pre_place[2] = safe_height

        q_pre_pick = self.robot.inverse_kinematics(pre_pick)
        q_pick     = self.robot.inverse_kinematics(pick_pos)
        q_pre_place = self.robot.inverse_kinematics(pre_place)
        q_place     = self.robot.inverse_kinematics(place_pos)

        # Sequence
        self._animate_segment(q_pre_pick, steps=60)
        self._animate_segment(q_pick, steps=40)

        carry = obj_idx

        self._animate_segment(q_pre_pick, steps=40, carry_index=carry)
        self._animate_segment(q_pre_place, steps=60, carry_index=carry)
        self._animate_segment(q_place, steps=40, carry_index=carry)

        # ----------------------------------------------------------
        # FIX: DETACH OBJECT *AT ACTUAL END-EFFECTOR LOCATION*
        # ----------------------------------------------------------
        final_joints = self.robot.forward_kinematics(q_place)
        final_ee = final_joints[-1]

        self.objects_pos[carry] = final_ee
        sc = self.objects_scatters[carry]
        sc._offsets3d = ([final_ee[0]], [final_ee[1]], [final_ee[2]])

        self.object_done[carry] = True
        # ----------------------------------------------------------

        self._animate_segment(q_pre_place, steps=40)


    # --------------------------------------------------------------
    def go_home(self):
        self._animate_segment(self.q_home, steps=60)


    def start(self):
        plt.show()


def run():
        sim = ManipulatorSimulation()
        sim.start()


if __name__ == "__main__":
    run()
