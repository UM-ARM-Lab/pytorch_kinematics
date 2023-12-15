#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 27.11.23
from typing import Collection, Optional

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import torch

from pytorch_kinematics import Transform3d


def visualize(frames: Transform3d, goal: Optional[torch.Tensor] = None, show: bool = False, **kwargs) -> plt.Axes:
    """
    Visualizes a sequence of frames.
    Args:
        frames: The forward kinematics transforms of a single robot.
        goal: The goal to visualize.
        show: Whether to show the plot.

    Returns: None
    """
    frames = np.vstack([f.get_matrix().cpu().detach().numpy() for f in frames])
    axis_size = np.max(frames[:, :3, 3]) - np.min(frames[:, :3, 3]) * .8
    center = np.mean(frames[:, :3, 3], axis=0)
    frame_scale = axis_size / 15
    ax = plt.figure(figsize=(12, 12)).add_subplot(projection='3d')
    frames = frames.reshape(-1, 4, 4)
    num_frames = frames.shape[0]
    draw_base(ax, scale=frame_scale, **kwargs)
    if goal is not None:
        draw_goal(ax, goal, scale=frame_scale, **kwargs)
    for i, frame in enumerate(frames):
        draw_frame(ax, frame, scale=frame_scale, **kwargs)
        if i < num_frames - 1:
            draw_link(ax, frame[:3, 3], frames[i + 1][:3, 3], **kwargs)
    ax.set_xlim(center[0] - axis_size, center[0] + axis_size)
    ax.set_ylim(center[1] - axis_size, center[1] + axis_size)
    ax.set_zlim(center[2] - axis_size, center[2] + axis_size)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if show:
        plt.show()
    return ax


def draw_base(ax: axes3d.Axes3D, scale: float = .1, **kwargs):
    """Draw a sphere of radius scale at the origin."""
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = scale * np.cos(u) * np.sin(v)
    y = scale * np.sin(u) * np.sin(v)
    z = scale * np.cos(v)
    ax.plot_wireframe(x, y, z, color='gray')


def draw_goal(ax: axes3d.Axes3D, goal: torch.Tensor, scale: float = .1, **kwargs):
    """Draw a sphere of radius scale at the origin."""
    u, v = np.mgrid[0:2 * np.pi:10j, 0:np.pi:5j]
    goal = np.squeeze(goal.cpu().detach().numpy())
    x = goal[0] + scale * np.cos(u) * np.sin(v)
    y = goal[1] + scale * np.sin(u) * np.sin(v)
    z = goal[2] + scale * np.cos(v)
    ax.plot_wireframe(x, y, z, color='red')


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
    ax.quiver(origin[0], origin[1], origin[2], x[0], x[1], x[2], length=scale, color='r', linewidths=3)
    ax.quiver(origin[0], origin[1], origin[2], y[0], y[1], y[2], length=scale, color='g', linewidths=3)
    ax.quiver(origin[0], origin[1], origin[2], z[0], z[1], z[2], length=scale, color='b', linewidths=3)


def draw_link(ax: axes3d.Axes3D, p0: np.ndarray, p1: np.ndarray, **kwargs):
    """
    Draws a line from p0 to p1.
    Args:
        ax: The axis to draw on.
        p0: The start point.
        p1: The end point.

    Returns: None
    """
    kwargs.setdefault('color', 'black')
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], linewidth=3, **kwargs)
