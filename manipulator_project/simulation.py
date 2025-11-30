import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from robot import ArticulatedRRRP


class ManipulatorSimulation:
    """
    Articulated 4-DOF manipulator performing pick-and-place of
    three objects from fixed floor positions onto a three-level shelf.

    UI:
        - Floor 1, Floor 2, Floor 3: run pick-&-place for that object
        - Reset: go to Home pose

    Also shows a visible Home reference point.
    """

    def __init__(self):
        # Robot model
        self.robot = ArticulatedRRRP()

        # Home configuration (arm along +X, mid-height)
        self.q_home = np.array([0.0, 0.0, 0.0, 0.35])
        self.q_current = self.q_home.copy()

        # Fixed floor object positions (three levels along front arc)
        self.floor_positions = np.array(
            [
                [0.35, -0.15, 0.02],  # Floor 1
                [0.40,  0.00, 0.02],  # Floor 2
                [0.35,  0.15, 0.02],  # Floor 3
            ]
        )

        # --- Shelf geometry ---
        # Put the shelf centered behind the objects in X,
        # and push it up so its back edge touches the "wall" at y = 0.7.
        self.shelf_x = 0.40          # roughly aligned with floor objects in X
        self.shelf_y = 0.52          # front of shelf; back will be at 0.52 + depth (0.18) = 0.70

        self.shelf_levels = np.array([0.25, 0.45, 0.65])  # Shelf 1,2,3 heights

        # Matching target positions on shelf
        self.shelf_positions = np.array(
            [
                [self.shelf_x, self.shelf_y, self.shelf_levels[0]],  # Shelf 1
                [self.shelf_x, self.shelf_y, self.shelf_levels[1]],  # Shelf 2
                [self.shelf_x, self.shelf_y, self.shelf_levels[2]],  # Shelf 3
            ]
        )

        # Track whether each object has already been placed
        self.object_done = [False, False, False]

        # Matplotlib figure / axes
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

        # Environment: floor, shelf, objects, home reference
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

        # UI buttons: only 4
        self._create_buttons()

    # ------------------------------------------------------------------
    # Environment drawing
    # ------------------------------------------------------------------
    def _draw_floor(self):
        # Simple rectangular floor plane
        x = np.linspace(-0.7, 0.7, 2)
        y = np.linspace(-0.5, 0.5, 2)
        xx, yy = np.meshgrid(x, y)
        zz = np.zeros_like(xx)
        self.ax.plot_surface(xx, yy, zz, color="lightgray", alpha=0.2)

    def _draw_shelf(self):
        """
        Proper looking shelf:
            - Two vertical posts
            - Three horizontal shelves (plates) at given z levels
        """
        width = 0.35
        depth = 0.18

        # Use self.shelf_x, self.shelf_y but keep the back edge on the 'wall' at y ~ 0.7
        x_left = self.shelf_x - width / 2
        x_right = self.shelf_x + width / 2
        y_front = self.shelf_y
        y_back = self.shelf_y + depth   # this will be ~0.70, touching the top boundary

        # Vertical posts (from floor to top shelf + bit)
        z_top = self.shelf_levels[-1] + 0.05
        for x_post in [x_left, x_right]:
            self.ax.plot(
                [x_post, x_post],
                [y_front, y_front],
                [0.0, z_top],
                color="dimgray",
                linewidth=3,
            )
            self.ax.plot(
                [x_post, x_post],
                [y_back, y_back],
                [0.0, z_top],
                color="dimgray",
                linewidth=3,
            )

        # Three shelf plates
        colors = ["lightblue", "peachpuff", "lightgreen"]
        for z, col in zip(self.shelf_levels, colors):
            xx = np.array([[x_left, x_right], [x_left, x_right]])
            yy = np.array([[y_front, y_front], [y_back, y_back]])
            zz = np.array([[z, z], [z, z]])
            self.ax.plot_surface(
                xx, yy, zz,
                color=col,
                alpha=0.7,
                edgecolor="k",
                linewidth=0.4,
            )

        # Label shelves
        for i, z in enumerate(self.shelf_levels, start=1):
            self.ax.text(
                x_right + 0.02,
                y_back,
                z + 0.02,
                f"Shelf {i}",
                fontsize=8,
            )

    def _create_objects(self):
        """Create three floor objects at fixed positions."""
        self.objects_pos = self.floor_positions.copy()
        self.objects_scatters = []

        colors = ["tab:blue", "tab:orange", "tab:green"]
        for i, p in enumerate(self.objects_pos):
            sc = self.ax.scatter(
                p[0], p[1], p[2],
                s=50, marker="s", color=colors[i]
            )
            self.objects_scatters.append(sc)
            self.ax.text(p[0], p[1], p[2] + 0.03, f"O{i+1}", fontsize=8)

    def _draw_home_reference(self):
        ee_home = self.robot.forward_kinematics(self.q_home)[-1]
        self.ax.scatter(
            ee_home[0], ee_home[1], ee_home[2],
            s=60, color="magenta", marker="o",
        )
        self.ax.text(
            ee_home[0],
            ee_home[1],
            ee_home[2] + 0.03,
            "Home ref",
            color="magenta",
            fontsize=8,
        )

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------
    def _create_buttons(self):
        # Only 4 buttons as requested
        names = ["Floor 1", "Floor 2", "Floor 3", "Reset"]
        self.buttons = {}
        start_y = 0.75
        dy = 0.08

        for i, name in enumerate(names):
            ax_btn = self.fig.add_axes([0.03, start_y - i * dy, 0.12, 0.06])
            btn = Button(ax_btn, name)

            if name == "Reset":
                btn.on_clicked(lambda event: self.go_home())
            else:
                idx = int(name.split()[-1]) - 1  # Floor N -> index
                btn.on_clicked(lambda event, k=idx: self.pick_and_place(k))

            self.buttons[name] = btn

    # ------------------------------------------------------------------
    # Animation helper
    # ------------------------------------------------------------------
    def _animate_segment(self, q_target, steps=60, carry_index=None):
        q_start = self.q_current.copy()
        for i in range(steps + 1):
            alpha = i / steps
            q = (1.0 - alpha) * q_start + alpha * q_target
            joints = self.robot.forward_kinematics(q)

            # Update robot line
            self.robot_line.set_data(joints[:, 0], joints[:, 1])
            self.robot_line.set_3d_properties(joints[:, 2])

            # If carrying an object, move it with the end-effector
            if carry_index is not None:
                ee = joints[-1]
                self.objects_pos[carry_index] = ee
                sc = self.objects_scatters[carry_index]
                sc._offsets3d = ([ee[0]], [ee[1]], [ee[2]])

            plt.pause(0.01)

        self.q_current = q_target.copy()

    # ------------------------------------------------------------------
    # Button actions
    # ------------------------------------------------------------------
    def pick_and_place(self, obj_idx: int):
        """
        Full sequence for one object:
            - From current pose -> over object
            - Lower to pick
            - Lift
            - Move over shelf
            - Lower to place
            - Leave object on shelf
            - Back up to safe height
        """

        # If already placed, do nothing (object stays on shelf)
        if self.object_done[obj_idx]:
            return

        pick_pos = self.floor_positions[obj_idx]
        place_pos = self.shelf_positions[obj_idx]

        safe_height = 0.75

        # Pre-pick: above object
        pre_pick = pick_pos.copy()
        pre_pick[2] = safe_height
        q_pre_pick = self.robot.inverse_kinematics(pre_pick)

        # Pick: at floor
        q_pick = self.robot.inverse_kinematics(pick_pos)

        # Pre-place: above shelf
        pre_place = place_pos.copy()
        pre_place[2] = safe_height
        q_pre_place = self.robot.inverse_kinematics(pre_place)

        # Place: at shelf level
        q_place = self.robot.inverse_kinematics(place_pos)

        # Sequence with animation
        self._animate_segment(q_pre_pick, steps=60)                    # move above object
        self._animate_segment(q_pick, steps=40)                        # go down to object

        # Attach object (carry)
        carry = obj_idx
        self._animate_segment(q_pre_pick, steps=40, carry_index=carry)   # lift
        self._animate_segment(q_pre_place, steps=60, carry_index=carry)  # move above shelf
        self._animate_segment(q_place, steps=40, carry_index=carry)      # down to shelf

        # Detach: leave object on shelf (stop carrying)
        self.objects_pos[carry] = place_pos
        sc = self.objects_scatters[carry]
        sc._offsets3d = ([place_pos[0]], [place_pos[1]], [place_pos[2]])

        # Mark as done so it stays on shelf
        self.object_done[carry] = True

        # Move back up, not carrying
        self._animate_segment(q_pre_place, steps=40, carry_index=None)

    def go_home(self):
        """Return to home configuration with animation."""
        self._animate_segment(self.q_home, steps=60)

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------
    def start(self):
        plt.show()


def run():
    sim = ManipulatorSimulation()
    sim.start()


if __name__ == "__main__":
    run()
