import asyncio
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src.pipeline.blue_team.monitoring import (
    add_basic_blue_team_monitor_to_pipeline,
    add_basic_blue_team_monitor_to_pipeline_sampling,
    add_basic_scalar_blue_team_monitor_to_pipeline,
)
from src.pipeline.pipeline import Pipeline, PipelineConfig
from src.pipeline.red_team.monitoring import add_red_team_sneakiness_steps_to_pipeline
from src.tools.dataloaders import load_multiple_solutions, load_solutions
from src.tools.path_utils import get_default_results_directory


def get_pipeline(
    strong_model,
    weak_model,
    num_problems=None,
    name=None,
    logprob=True,
    bon=1,
    temperature=0,
):
    config = PipelineConfig(
        name,
        anthropic_num_threads=40,
        openai_fraction_rate_limit=0.99,
        num_problems=num_problems,
        use_cache=True,
    )

    pipeline = Pipeline(config)

    get_innocent_solutions = pipeline.add_load_data_step(
        "get_innocent_solutions",
        load_solutions,
        get_default_results_directory()
        / "trusted_monitoring-newredteam"
        / "01-get_solutions"
        / strong_model,
    )

    get_backdoored_solutions = pipeline.add_load_data_step(
        "get_backdoored_solutions_1",
        load_multiple_solutions,
        get_default_results_directory()
        / "trusted_monitoring-newredteam"
        / "37-final_backdoors"
        / f"{strong_model}+gpt-3.5-turbo-1106",
    )

    get_backdoored_solutions_2 = pipeline.add_load_data_step(
        "get_backdoored_solutions_2",
        load_multiple_solutions,
        get_default_results_directory()
        / "trusted_editing-newredteam"
        / "37-final_backdoors"
        / f"{strong_model}+gpt-3.5-turbo-1106",
    )

    def merge(x, y):
        merged = {}
        for key in x.keys():
            merged[key] = x[key]
        # for key in y.keys():
        #     if key in x.keys():
        #         merged[key].extend(y[key])
        #     else:
        #         merged[key] = y[key]
        return {
            f"{key}-{i}": value
            for key, value_list in merged.items()
            for (i, value) in enumerate(value_list)
        }

    merged_backdoors = pipeline.add_transformation_step(
        "merged_backdoors",
        merge,
        dependencies=[get_backdoored_solutions, get_backdoored_solutions],
        strong_model=strong_model,
        weak_model=weak_model,
    )

    if not logprob:  # get scalar (1-10)
        get_suspiciousness_innocent = add_basic_scalar_blue_team_monitor_to_pipeline(
            pipeline,
            strong_model,
            weak_model,
            get_innocent_solutions,
            name="innocent",
            bon=bon,
            temperature=temperature,
        )
        get_suspiciousness_backdoors = add_basic_scalar_blue_team_monitor_to_pipeline(
            pipeline,
            strong_model,
            weak_model,
            merged_backdoors,
            name="backdoors",
            bon=bon,
            temperature=temperature,
        )
    else:  # get logprpob
        if "claude" in weak_model:
            get_suspiciousness_innocent = (
                add_basic_blue_team_monitor_to_pipeline_sampling(
                    pipeline,
                    strong_model,
                    weak_model,
                    get_innocent_solutions,
                    name="innocent",
                    bon=bon,
                    temperature=temperature,
                )
            )
            get_suspiciousness_backdoors = (
                add_basic_blue_team_monitor_to_pipeline_sampling(
                    pipeline,
                    strong_model,
                    weak_model,
                    merged_backdoors,
                    name="backdoors",
                    bon=bon,
                    temperature=temperature,
                )
            )
        else:
            get_suspiciousness_innocent = add_basic_blue_team_monitor_to_pipeline(
                pipeline,
                strong_model,
                weak_model,
                get_innocent_solutions,
                name="innocent",
            )
            get_suspiciousness_backdoors = add_basic_blue_team_monitor_to_pipeline(
                pipeline, strong_model, weak_model, merged_backdoors, name="backdoors"
            )

    get_sneakiness_innocent = add_red_team_sneakiness_steps_to_pipeline(
        pipeline, strong_model, weak_model, get_innocent_solutions, name="innocent"
    )
    get_sneakiness_backdoors = add_red_team_sneakiness_steps_to_pipeline(
        pipeline, strong_model, weak_model, get_innocent_solutions, name="backdoors"
    )

    results = asyncio.run(pipeline.run())
    # Extract suspiciousness scores for innocent solutions
    innocent_scores = [
        result["suspiciousness_innocent"]
        for result in results[get_suspiciousness_innocent.name].values()
        if result.get("suspiciousness_innocent") is not None
    ]

    innocent_sneakiness = [
        result["sneakiness"]
        for result in results[get_sneakiness_innocent.name].values()
        if result.get("sneakiness") is not None
    ]

    # Extract suspiciousness scores for backdoored solutions
    backdoored_scores = [
        result["suspiciousness_backdoors"]
        for result in results[get_suspiciousness_backdoors.name].values()
        if result.get("suspiciousness_backdoors") is not None
    ]

    backdoored_sneakiness = [
        result["sneakiness"]
        for result in results[get_sneakiness_backdoors.name].values()
        if result.get("sneakiness") is not None
    ]

    return (
        pipeline,
        innocent_scores,
        backdoored_scores,
        innocent_sneakiness,
        backdoored_sneakiness,
    )


def plot_results(
    innocent_scores,
    backdoored_scores,
    innocent_sneakiness,
    backdoored_sneakiness,
    name,
    strong_model,
    weak_model,
    logprob=True,
):
    save_dir = f"{get_default_results_directory()}/{name}/merged_results/{strong_model}+{weak_model}"
    os.makedirs(save_dir, exist_ok=True)

    if not logprob:
        innocent_scores = [i / 10 for i in innocent_scores]
        backdoored_scores = [i / 10 for i in backdoored_scores]

    # Calculate mean and standard deviation
    innocent_mean = np.mean(innocent_scores)
    innocent_std = np.std(innocent_scores)
    backdoored_mean = np.mean(backdoored_scores)
    backdoored_std = np.std(backdoored_scores)

    innocent_sneakiness_mean, innocent_sneakiness_std = np.mean(
        innocent_sneakiness
    ), np.std(innocent_sneakiness)
    backdoored_sneakiness_mean, backdoored_sneakiness_std = np.mean(
        backdoored_sneakiness
    ), np.std(backdoored_sneakiness)

    all_scores = np.concatenate([innocent_scores, backdoored_scores])
    all_sneakiness_scores = np.concatenate([innocent_sneakiness, backdoored_sneakiness])

    labels = np.concatenate(
        [np.zeros(len(innocent_scores)), np.ones(len(backdoored_scores))]
    )

    # Print statistics
    print(
        f"Innocent solutions - Mean: {innocent_mean:.2f}, Std Dev: {innocent_std:.2f}, Mean Sneakiness: {innocent_sneakiness_mean:.2f}, Std Dev: {innocent_sneakiness_std:.2f}"
    )
    print(
        f"Backdoored solutions - Mean: {backdoored_mean:.2f}, Std Dev: {backdoored_std:.2f}, Mean Sneakiness: {backdoored_sneakiness_mean:.2f}, Std Dev: {backdoored_sneakiness_std:.2f}"
    )

    # Create histogram
    # plt.figure(figsize=(10, 6))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    ax1.hist(
        innocent_scores,
        bins=20,
        alpha=0.5,
        color="blue",
        label="Innocent",
        density=True,
    )
    ax1.hist(
        backdoored_scores,
        bins=20,
        alpha=0.5,
        color="red",
        label="Backdoored",
        density=True,
    )

    ax1.set_xlabel("Suspiciousness Score")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Suspiciousness Scores")
    ax1.legend()

    # Add vertical lines for means
    ax1.axvline(innocent_mean, color="blue", linestyle="dashed", linewidth=2)
    ax1.axvline(backdoored_mean, color="red", linestyle="dashed", linewidth=2)

    ax2.hist(
        innocent_sneakiness,
        bins=20,
        alpha=0.5,
        color="blue",
        label="Innocent",
        density=True,
    )
    ax2.hist(
        backdoored_sneakiness,
        bins=20,
        alpha=0.5,
        color="red",
        label="Backdoored",
        density=True,
    )

    ax2.set_xlabel("Sneakiness Score")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of Sneakiness Scores")
    ax2.legend()

    # Add vertical lines for means
    ax1.axvline(innocent_sneakiness_mean, color="blue", linestyle="dashed", linewidth=2)
    ax1.axvline(
        backdoored_sneakiness_mean, color="red", linestyle="dashed", linewidth=2
    )

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{name}.png", dpi=1000)
    plt.close()


if __name__ == "__main__":
    bon = 1
    temp = 0

    name = f"sneakiness_ablation-gpt-3.5-turbo-logprob-bon{bon}-temperature{temp}"
    strong_model = "gpt-4-0125-preview"
    weak_model = "gpt-3.5-turbo-1106"

    (
        pipeline,
        innocent_scores,
        backdoored_scores,
        innocent_sneakiness,
        backdoored_sneakiness,
    ) = get_pipeline(
        strong_model=strong_model,
        weak_model=weak_model,
        num_problems=200,
        name=name,
        logprob=True,
        bon=bon,
        temperature=temp,
    )
    print(innocent_sneakiness)
    print("=" * 100)
    print(backdoored_sneakiness)
    plot_results(
        innocent_scores,
        backdoored_scores,
        innocent_sneakiness,
        backdoored_sneakiness,
        pipeline.config.name,
        strong_model,
        weak_model,
    )
