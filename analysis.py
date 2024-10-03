"""Functions used to plot charts in the Thesis."""

from collections import Counter
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def plot_wider_chart_with_legend_outside(new_data):
    # Used to reproduce the scatter plot from chapter 4
    plt.figure(figsize=(10, 6))  # Increase the width of the plot

    # Define marker mapping for plan_format (shape)
    plan_format_mapping = {"json": "o", "gml": "s"}  # Circle for JSON, Square for GML
    finetuning_colors = {
        "baseline": "#0072B2",
        "tool_bert": "#D55E00",
        "none": "#8B8B8B",
    }  # Color-blind friendly + 'none' case

    # First scatter plot: color for finetune strategy and shape for plan_format
    for finetuning in new_data["finetuning_strategy"].unique():
        for plan_format in new_data["plan_format"].unique():
            subset = new_data[
                (new_data["finetuning_strategy"] == finetuning)
                & (new_data["plan_format"] == plan_format)
            ]
            plt.scatter(
                x=subset["success_rate"],
                y=subset["mean_alignment_all"],
                color=finetuning_colors.get(
                    finetuning, "#8B8B8B"
                ),  # Use gray for any unexpected values
                marker=plan_format_mapping[plan_format],
                s=100,
                label=f"{finetuning} | {plan_format}",
            )

    # Set X and Y axis labels
    plt.xlabel("Success Rate")
    plt.ylabel("Alignment Score")

    # Adjust X and Y axis limits with padding
    plt.xlim(0 - 0.05, 1 + 0.05)  # Adding padding on X-axis
    plt.ylim(0 - 0.05, 1 + 0.05)  # Adding padding on Y-axis

    # Move the legend outside the plot space
    plt.legend(
        title="Finetuning Strategy | Plan Format",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter


def errors_chart():
    # Prepare data for plotting
    counter = Counter(
        {
            "PLAN_GENERATION": 2038,
            "MEMORY": 568,
            "INVALID_ARGUMENT": 476,
            "TOOL_SIGNATURE": 224,
            "ARGUMENT_VALIDATION": 214,
            "UNEXPECTED": 0,
        }
    )

    # Bar chart with Seaborn from  Counter
    labels = list(counter.keys())
    values = list(counter.values())

    # Create a DataFrame for Seaborn
    df = pd.DataFrame({"Labels": labels, "Values": values})

    # Sort DataFrame by the order of Labels if needed
    df["Labels"] = pd.Categorical(df["Labels"], categories=labels, ordered=True)
    df = df.sort_values("Labels")  # Sort by Labels

    # Create a bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Labels", y="Values", data=df, palette="viridis")

    # Add titles and labels
    plt.title("Bar Chart of Values by Labels", fontsize=16)
    plt.xlabel("Labels", fontsize=14)
    plt.ylabel("Values", fontsize=14)

    # Rotate x labels for better readability
    plt.xticks(rotation=45)

    # Save the plot as an image file
    plt.savefig("bar_chart_values_by_labels.png", bbox_inches="tight")


if __name__ == "__main__":
    errors_chart()
