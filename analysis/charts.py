from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file
file_path = Path(__file__).parent / "data" / "benchmark" / "metrics.csv"
data = pd.read_csv(file_path)

# Define a color palette for 'is_online_strategy'
color_palette = {"Online": "blue", "Offline": "orange"}

# Define marker styles for 'generation_strategy_name'
marker_styles = {"cot": "o", "react": "s", "blind": "^"}

# Replace 'is_online_strategy' values with 'Online' and 'Offline' for the legend
data["is_online_strategy"] = data["is_online_strategy"].map(
    {True: "Online", False: "Offline"}
)

# Columns for X and Y axes
x_columns = [
    "num_demonstrations",
    "error_feedback_strategy",
    "retry_times",
    "mean_tokens",
]
y_columns = ["mean_alignment_success", "mean_alignment_failed", "success_rate"]

# Replace string-based enums with integers to allow better plotting for 'error_feedback_strategy'
data["error_feedback_strategy"] = data["error_feedback_strategy"].map(
    {"NO_FEEDBACK": 0, "ERROR_TYPE": 1}
)

# Create a figure with subplots for each combination of X and Y
fig, axes = plt.subplots(len(x_columns), len(y_columns), figsize=(18, 12))
plt.subplots_adjust(hspace=0.5, wspace=0.4)

# Iterate through each pair of X and Y
for i, x_col in enumerate(x_columns):
    for j, y_col in enumerate(y_columns):
        ax = axes[i, j]

        # Plot each point with color and marker style based on 'is_online_strategy' and 'generation_strategy_name'
        for gen_strategy, marker in marker_styles.items():
            for is_online, color in color_palette.items():
                subset = data[
                    (data["generation_strategy_name"] == gen_strategy)
                    & (data["is_online_strategy"] == is_online)
                ]
                ax.scatter(
                    subset[x_col],
                    subset[y_col],
                    label=f"{gen_strategy}, {is_online}",
                    c=color,
                    marker=marker,
                    edgecolors="k",
                    s=80,
                    alpha=0.7,
                )

        # Set labels and title for each subplot
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{x_col} vs {y_col}")

# Add a global legend outside the plot
handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper right",
    title="Legend (Generation Strategy, Online/Offline)",
    bbox_to_anchor=(1.15, 1),
)

plt.show()
