#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 27.11.23
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np


def visualize(frames: np.ndarray):
    """
    Visualizes a sequence of frames.
    Args:
        frames: An nx4x4 array of frame placements. Assumes the first frame is the base frame (no implicit world frame
        is assumed).

    Returns: None
    """
    axis_size = np.max(frames[:, :3, 3]) - np.min(frames[:, :3, 3]) * .8
    center = np.mean(frames[:, :3, 3], axis=0)
    frame_scale = axis_size / 15
    ax = plt.figure(figsize=(12, 12)).add_subplot(projection='3d')
    frames = frames.reshape(-1, 4, 4)
    num_frames = frames.shape[0]
    for i, frame in enumerate(frames):
        draw_frame(ax, frame, scale=frame_scale)
        if i < num_frames - 1:
            draw_link(ax, frame[:3, 3], frames[i + 1][:3, 3])
    ax.set_xlim(center[0] - axis_size, center[0] + axis_size)
    ax.set_ylim(center[1] - axis_size, center[1] + axis_size)
    ax.set_zlim(center[2] - axis_size, center[2] + axis_size)
    plt.show()


def draw_frame(ax: axes3d.Axes3D, frame: np.ndarray, scale: float = .1):
    """
    Draws a frame.
    Args:
        ax: The axis to draw on.
        frame: The frame placement.
        scale: The length of the frame axes.

    Returns: None
    """
    origin = frame[:3, 3]
    x = frame[:3, 0]
    y = frame[:3, 1]
    z = frame[:3, 2]
    ax.quiver(origin[0], origin[1], origin[2], x[0], x[1], x[2], length=scale, color='r')
    ax.quiver(origin[0], origin[1], origin[2], y[0], y[1], y[2], length=scale, color='g')
    ax.quiver(origin[0], origin[1], origin[2], z[0], z[1], z[2], length=scale, color='b')


def draw_link(ax: axes3d.Axes3D, p0: np.ndarray, p1: np.ndarray):
    """
    Draws a line from p0 to p1.
    Args:
        ax: The axis to draw on.
        p0: The start point.
        p1: The end point.

    Returns: None
    """
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color='black', linewidth=3)
