# simulation.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D
from robot import ArticulatedRRPR
import time

class Simulator:
    def __init__(self, robot=None):
        self.robot = robot if robot is not None else ArticulatedRRPR()

        # default joint pose
        self.q = np.array([0.0, 0.0, 0.0, 0.0])

        # define targets: floor + 3 shelves (different elevations)
        self.targets = {
            'Floor': np.array([0.5, 0.0, 0.0]),
            'Shelf 1': np.array([0.3, 0.3, 0.4]),
            'Shelf 2': np.array([0.5, -0.2, 0.6]),
            'Shelf 3': np.array([0.2, 0.4, 0.8]),
        }

        # create figure & axes
        self.fig = plt.figure(figsize=(10,7))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_box_aspect([1,1,0.8])

        # UI buttons area
        axcolor = 'lightgoldenrodyellow'
        self.ax_floor = plt.axes([0.02, 0.85, 0.12, 0.06])
        self.ax_s1 = plt.axes([0.02, 0.76, 0.12, 0.06])
        self.ax_s2 = plt.axes([0.02, 0.67, 0.12, 0.06])
        self.ax_s3 = plt.axes([0.02, 0.58, 0.12, 0.06])

        self.btn_floor = Button(self.ax_floor, 'Floor')
        self.btn_s1 = Button(self.ax_s1, 'Shelf 1')
        self.btn_s2 = Button(self.ax_s2, 'Shelf 2')
        self.btn_s3 = Button(self.ax_s3, 'Shelf 3')

        self.btn_floor.on_clicked(lambda event: self.move_to('Floor'))
        self.btn_s1.on_clicked(lambda event: self.move_to('Shelf 1'))
        self.btn_s2.on_clicked(lambda event: self.move_to('Shelf 2'))
        self.btn_s3.on_clicked(lambda event: self.move_to('Shelf 3'))

        # initial plot
        self.draw_scene()
        plt.subplots_adjust(left=0.18, right=0.98, top=0.95, bottom=0.05)

    def draw_scene(self):
        self.ax.cla()
        # axis labels
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_xlim(-0.8, 0.8)
        self.ax.set_ylim(-0.8, 0.8)
        self.ax.set_zlim(0, 1.2)

        # draw shelves as rectangular planes for visualization
        self._draw_shelves()

        # draw robot at current q
        self._draw_robot(self.q)

        # draw target markers
        for name, t in self.targets.items():
            self.ax.scatter([t[0]], [t[1]], [t[2]], marker='o')
            self.ax.text(t[0], t[1], t[2]+0.03, name, fontsize=8)

        self.fig.canvas.draw_idle()

    def _draw_shelves(self):
        # simple shelf representation: at z=0.4, 0.6, 0.8
        for z, x0 in [(0.4, 0.0), (0.6, 0.0), (0.8, 0.0)]:
            # a small rectangular plane
            xs = np.array([-0.6, 0.6, 0.6, -0.6])
            ys = np.array([-0.2, -0.2, 0.6, 0.6])
            zs = np.array([z, z, z, z])
            self.ax.plot_trisurf(xs, ys, zs, alpha=0.06, shade=False)

    def _draw_robot(self, q):
        # compute joint positions for stick figure
        theta1, theta2, d3, theta4 = q
        base = np.array([0.0, 0.0, 0.0])
        p0 = base

        # joint1 position after link a1
        p1 = p0 + np.array([self.robot.a1 * np.cos(theta1),
                            self.robot.a1 * np.sin(theta1),
                            0.0])

        # joint2 (after a2)
        p2 = p1 + np.array([self.robot.a2 * np.cos(theta1 + theta2),
                            self.robot.a2 * np.sin(theta1 + theta2),
                            0.0])

        # wrist (a4), and then prismatic contributes to z
        p3 = p2 + np.array([self.robot.a4 * np.cos(theta1 + theta2 + theta4),
                            self.robot.a4 * np.sin(theta1 + theta2 + theta4),
                            d3])

        # plot links
        xs = [p0[0], p1[0], p2[0], p3[0]]
        ys = [p0[1], p1[1], p2[1], p3[1]]
        zs = [p0[2], p1[2], p2[2], p3[2]]
        self.ax.plot(xs, ys, zs, '-o', linewidth=3, markersize=6)

        # annotate joints
        self.ax.text(p0[0], p0[1], p0[2]+0.02, 'Base', fontsize=8)
        self.ax.text(p1[0], p1[1], p1[2]+0.02, 'J1_end', fontsize=8)
        self.ax.text(p2[0], p2[1], p2[2]+0.02, 'J2_end', fontsize=8)
        self.ax.text(p3[0], p3[1], p3[2]+0.02, 'EE', fontsize=8)

    def move_to(self, name):
        target = self.targets[name]
        print(f"Requested move to {name}: {target}")
        # compute IK
        q_target = self.robot.inverse_kinematics(target)
        print("IK result:", q_target)
        # animate trajectory
        self._animate_to(q_target)

    def _animate_to(self, q_target, steps=80, pause=0.01):
        q_start = self.q.copy()
        for i in range(1, steps+1):
            alpha = i / steps
            q_interp = (1-alpha) * q_start + alpha * q_target
            # For robustness, clip d3 to allowed range
            q_interp[2] = np.clip(q_interp[2], self.robot.d3_min, self.robot.d3_max)
            self.q = q_interp
            self.draw_scene()
            plt.pause(pause)
        # final set
        self.q = q_target
        self.draw_scene()

def run():
    sim = Simulator()
    plt.show()

if __name__ == "__main__":
    run()
