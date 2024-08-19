import os
import numpy as np
import seaborn
from matplotlib import pyplot as plt
from stable_baselines3.common.monitor import LoadMonitorResultsError, load_results
from stable_baselines3.common.results_plotter import X_EPISODES, X_TIMESTEPS, X_WALLTIME, ts2xy, window_func

# Activate seaborn
seaborn.set()
ALGO = ['ppo', 'sac', 'a2c']

def plot_train(
    algos,
    envs,
    exp_folder,
    figsize=[10, 6],
    fontsize=14,
    max_timesteps=1e6,
    args_x_axis="steps",
    args_y_axis="reward",
    episode_window=100,
    save_path="plots"
):
    x_axis_map = {
        "steps": X_TIMESTEPS,
        "episodes": X_EPISODES,
        "time": X_WALLTIME,
    }
    x_axis = x_axis_map[args_x_axis]
    x_label_map = {
        "steps": "Timesteps",
        "episodes": "Episodes",
        "time": "Walltime (in hours)",
    }
    x_label = x_label_map[args_x_axis]

    y_axis_map = {
        "success": "is_success",
        "reward": "r",
        "length": "l",
    }
    y_axis = y_axis_map[args_y_axis]
    y_label_map = {
        "success": "Training Success Rate",
        "reward": "Training Episodic Reward",
        "length": "Training Episode Length",
    }
    y_label = y_label_map[args_y_axis]

    # Ensure save_path directory exists
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=figsize)
    plt.title(f"Training Performance ({args_y_axis.capitalize()})", fontsize=fontsize)
    plt.xlabel(f"{x_label}", fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)

    for algo in algos:
        log_path = os.path.join(exp_folder, algo)
        dirs = []

        for env in envs:
            # Sort by last modification
            entries = sorted(os.scandir(log_path), key=lambda entry: entry.stat().st_mtime)
            dirs.extend(entry.path for entry in entries if env in entry.name and entry.is_dir())

        for folder in dirs:
            try:
                data_frame = load_results(folder)
            except LoadMonitorResultsError:
                continue
            if max_timesteps is not None:
                data_frame = data_frame[data_frame.l.cumsum() <= max_timesteps]
            try:
                y = np.array(data_frame[y_axis])
            except KeyError:
                print(f"No data available for {folder}")
                continue
            x, _ = ts2xy(data_frame, x_axis)

            # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
            if x.shape[0] >= episode_window:
                # Compute and plot rolling mean with window of size episode_window
                x, y_mean = window_func(x, y, episode_window, np.mean)
                plt.plot(x, y_mean, linewidth=2, label=f"{algo}-{folder.split('/')[-1]}")

    plt.legend()
    plt.tight_layout()

    # Save plot as SVG file
    save_filename = os.path.join(save_path, f"training_performance_{args_y_axis}_{args_x_axis}.svg")
    plt.savefig(save_filename, format='svg')
    plt.show()

if __name__ == "__main__":
    plot_train(
        algos=ALGO,
        envs=["energy_net-v0"],
        exp_folder="Models/Original_no_first",
        figsize=[10, 6],
        fontsize=14,
        max_timesteps=1e6,
        args_x_axis="steps",
        args_y_axis="reward",
        episode_window=48,
        save_path="plots"  # Directory where plots will be saved
    )
