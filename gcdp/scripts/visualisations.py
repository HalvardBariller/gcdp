"""This script contains visualisation functions for the GCDP project."""

import matplotlib as mpl
from matplotlib import cm, gridspec
import matplotlib.pyplot as plt
import numpy as np


def goal_map_visualisation(
    goal_pose,
    achieved_goals,
    num_refinement,
    num_rollout,
    save_path=None,
    save_fig=False,
    cmap="coolwarm",
):
    """Visualisation of the behavioral goal vs. the trajectory of achieved goals during a rollout.

    Specific to Push-T task (width/height = 680, action space = 512).
    Args:
        - goal_pose (array): the goal to reach
        - achieved_goals (list of arrays): the achieved goals during the rollout
        - num_refinement (int): the index of refinement
        - num_rollout (int): the index of rollout
        - save_path (str): the path to save the figure
        - save_fig (bool): whether to save the figure
    """
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])

    ax0.plot(goal_pose[0] / 512 * 680, goal_pose[1] / 512 * 680, "ro")
    ax0.set_xlim([0, 680])
    ax0.set_ylim([680, 0])
    ax0.set_aspect("equal")
    ax0.set_title("Behavioral goal used for rollout generation")
    num_goals = len(achieved_goals)
    # colors = [
    #     (i / num_goals, 0, 1.0 - i / num_goals) for i in range(num_goals)
    # ]
    colors = plt.get_cmap(cmap, num_goals)
    for i, achieved_goal in enumerate(achieved_goals):
        ax1.plot(
            achieved_goal[0] / 512 * 680,
            achieved_goal[1] / 512 * 680,
            color=colors(i, alpha=0.5),
            marker="o",
        )
    ax1.set_xlim([0, 680])
    ax1.set_ylim([680, 0])
    ax1.set_aspect("equal")
    ax1.set_title("Trajectory of achieved goals during rollout")

    # Create colorbar as a legend
    norm = mpl.colors.Normalize(vmin=0, vmax=num_goals - 1)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Add the colorbar to the plot
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label("Gradient Scale")
    cbar.set_ticks([0, num_goals - 1])
    cbar.set_ticklabels(["Start: t=0", "End: t=H"])

    fig.suptitle(
        f"Goal map visualisation (refinement={num_refinement}, rollout={num_rollout})",
        fontsize=16,
    )
    name = f"goal_map_visualisation_refinement{num_refinement}_rollout{num_rollout}.png"
    if save_fig:
        (
            fig.savefig(save_path + name)
            if save_path
            else ValueError("Please provide a save path.")
        )
    else:
        return fig, name


def aggregated_goal_map_visualisation(
    behavioral_achieved_goals: list,
    cmap: str = "viridis",
):
    """Visualisation of the aggregated behavioral goals vs. achieved goals through policy refinements.

    Specific to Push-T task (width/height = 680, action space = 512).
    Args:
        - behavioral_achieved_goals (list of list of arrays): the achieved goals during the rollout
    """
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])

    num_refinement = len(behavioral_achieved_goals)
    colors = plt.get_cmap(cmap, num_refinement)
    for i, rollout_goals in enumerate(behavioral_achieved_goals):
        behavioral = rollout_goals["behavioral_goal"]
        achieved = rollout_goals["achieved_goals"]
        ax0.plot(
            behavioral[0] / 512 * 680,
            behavioral[1] / 512 * 680,
            color=colors(i),
            marker="o",
        )
        for achieved_goal in achieved:
            ax1.plot(
                achieved_goal[0] / 512 * 680,
                achieved_goal[1] / 512 * 680,
                color=colors(i, alpha=0.5),
                marker="o",
            )
    ax0.set_xlim([0, 680])
    ax0.set_ylim([680, 0])
    ax0.set_aspect("equal")
    ax0.set_title("Behavioral goals used for rollout generations")
    ax1.set_xlim([0, 680])
    ax1.set_ylim([680, 0])
    ax1.set_aspect("equal")
    ax1.set_title("Achieved goals during rollouts")

    # Create colorbar as a legend
    norm = mpl.colors.Normalize(vmin=0, vmax=num_refinement - 1)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # Add the colorbar to the plot
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label("Gradient Scale")
    cbar.set_ticks([0, num_refinement - 1])
    cbar.set_ticklabels(["Start: p=1", f"End: p={num_refinement}"])

    name = f"aggregated_goal_map_visualisation_refinement{num_refinement}.png"
    fig.suptitle(
        f"Aggregated goal map visualisation obtained after p={num_refinement} policy refinements",
        fontsize=16,
    )

    return fig, name
