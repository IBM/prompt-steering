from sklearn.model_selection import train_test_split
from scipy import stats, special
from scipy.stats import wasserstein_distance
from scipy.optimize import minimize
from scipy.stats import beta

import matplotlib.pyplot as plt
import statistics
import numpy as np
import os.path
import pandas as pd
import json


from prompt_steerability.utils.data_utils import save_to_json


def filter_data(df, threshold, min_count=50):
    """
    Filters dataset based on given threshold and min_count to contain exactly min_count examples for each
    direction of each persona dimension. Prunes excess samples to ensure all dimensions have the same count.
    Raises ValueError if any persona_dim and direction combination does not have exactly min_count values after filtering.
    """

    df_filtered = df[df['label_confidence'] >= threshold].copy()
    grouped = df_filtered.groupby(['persona_dim', 'direction']).size().unstack(fill_value=0)
    min_counts = grouped.min(axis=1)
    valid_dims = min_counts[min_counts >= min_count].index.tolist()

    if not valid_dims:
        return pd.DataFrame(), []

    filtered_data = []
    for dim in valid_dims:
        dim_data = df_filtered[df_filtered['persona_dim'] == dim]
        pos_samples = dim_data[dim_data['direction'] == 'positive']
        neg_samples = dim_data[dim_data['direction'] == 'negative']

        filtered_data.append(pos_samples.sample(min_count))
        filtered_data.append(neg_samples.sample(min_count))

    result_df = pd.concat(filtered_data, ignore_index=True)

    final_counts = result_df.groupby(['persona_dim', 'direction']).size().unstack(fill_value=0)

    return result_df, valid_dims


def chunk_interval(budget, chunks, start=1):
    if start > budget:
        raise ValueError("start value cannot be greater than budget")

    if chunks >= budget - start + 1:
        return list(range(1, start + 1)) + list(range(start + 1, budget + 1))

    if chunks < 2:
        return list(range(1, start + 1)) + [budget]

    if chunks == 2:
        return list(range(1, start + 1)) + [budget]

    result = list(range(1, start + 1))

    remaining_chunks = chunks - len(result)
    remaining_interval = budget - start

    if remaining_chunks > 1:
        step = remaining_interval / (remaining_chunks - 1)
        for i in range(1, remaining_chunks):
            next_value = round(start + i * step)
            if next_value > result[-1]:
                result.append(next_value)
            else:
                result.append(result[-1] + 1)

    if result[-1] != budget:
        result.append(budget)

    return result


def get_data_splits(data,
                    profiling_size,
                    stratify_by=None):
    """
    Extracts steering and profiling data for each persona dimension from the provided dataset.

    """

    if stratify_by is not None:

        # temp - prune data down due to context length limitations of models
        # todo: resolve (limit to max context length of models under evaluation)
        _, data = train_test_split(data,
                                   test_size=0.1,
                                   # shuffle=False,
                                   stratify=stratify_by)
        stratify_by = data['stratify_by']
        # / temp

        test_size = round(profiling_size * len(data))
        data_steering, data_profiling = train_test_split(data,
                                                         test_size=test_size,
                                                         # shuffle=False,
                                                         stratify=stratify_by)
        return data_steering, data_profiling

    else:

        pass


def get_max_steered_profile_params(profiling_data, belief_update, direction):

    if direction == "positive":
        profiling_data['answer'] = profiling_data['direction'].apply(lambda x: 'yes' if x == 'positive' else 'no')
    elif direction == "negative":
        profiling_data['answer'] = profiling_data['direction'].apply(lambda x: 'no' if x == 'positive' else 'yes')
    else:
        raise Exception("direction not valid")

    return belief_update(profiling_data.to_dict('records'))


def compute_steerability_indices(profiles, profiling_data, belief_update, steering_budgets, distance_metric="wasserstein"):

    def compute_distance(dist1, dist2):
        if distance_metric == "wasserstein":
            return wasserstein_distance(dist1, dist2)

    x = np.linspace(0, 1, 200)

    base_params = []
    steerability_indices = []
    for trial, trial_profiles in enumerate(profiles):

        base_profile_params = trial_profiles['base']
        base_profile = stats.beta.cdf(x, base_profile_params['alpha'], base_profile_params['beta'])
        base_params.append(base_profile_params)

        steered_profiles = trial_profiles['steered']

        trial_steerability_indices = []
        for steering_budget_idx, steering_budget in enumerate(steering_budgets):

            max_steered_profile, profile_params_trials = {}, {}
            for direction in ['positive', 'negative']:
                max_steered_profile_params = get_max_steered_profile_params(profiling_data=profiling_data[trial],
                                                                            belief_update=belief_update,
                                                                            direction=direction)
                max_steered_profile[direction] = stats.beta.cdf(x,
                                                                max_steered_profile_params['alpha'],
                                                                max_steered_profile_params['beta'])
                profile_params_trials[direction] = steered_profiles[steering_budget_idx][direction]

            norm = wasserstein_distance(max_steered_profile['positive'], max_steered_profile['negative'])

            indices = {}
            for direction in ['positive', 'negative']:
                indices_by_trial = []
                for profile_params_trial in profile_params_trials[direction]:
                    profile_trial = stats.beta.cdf(x, profile_params_trial['alpha'], profile_params_trial['beta'])
                    distance = wasserstein_distance(base_profile, max_steered_profile[direction]) - \
                               wasserstein_distance(profile_trial, max_steered_profile[direction])
                    ratio = distance / norm
                    indices_by_trial.append(ratio)
                indices[direction] = {
                    'mean': statistics.mean(indices_by_trial),
                    'stdev': statistics.stdev(indices_by_trial)
                }

            trial_steerability_indices.append(indices)

        steerability_indices.append(trial_steerability_indices)

    return _combine_stats_indices(steerability_indices), _combine_stats_beta_params(base_params)


def _combine_stats_indices(_combine_stats_indices):
    return [{
        key: {
            'mean': statistics.mean(trial[i][key]['mean'] for trial in _combine_stats_indices),
            'stdev': statistics.stdev((trial[i][key]['mean'] for trial in _combine_stats_indices), statistics.mean(trial[i][key]['mean'] for trial in _combine_stats_indices)
            )
        } for key in ['positive', 'negative']
    } for i in range(len(_combine_stats_indices[0]))]


def _combine_stats_beta_params(base_params):
    means = []
    variances = []

    for params in base_params:
        alpha = params['alpha']
        beta = params['beta']
        mean = alpha / (alpha + beta)
        variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        means.append(mean)
        variances.append(variance)

    avg_mean = np.mean(means)
    combined_std = np.sqrt(np.mean(variances))

    return {
        'mean': avg_mean,
        'std': combined_std
    }


def plot_indices(indices, steering_budgets, dimension, model_name):
    budget_lim = len(indices)

    pos_means = [item['positive']['mean'] for item in indices]
    pos_stdevs = [item['positive']['stdev'] for item in indices]
    neg_means = [-item['negative']['mean'] for item in indices]
    neg_stdevs = [item['negative']['stdev'] for item in indices]

    plt.figure(figsize=(3.75, 3.75))

    plt.errorbar(steering_budgets, pos_means, yerr=pos_stdevs, fmt='g.', capsize=3,
                 ecolor='g', elinewidth=1, capthick=1, alpha=0.7, label='positive')
    plt.errorbar(steering_budgets, neg_means, yerr=neg_stdevs, fmt='r.', capsize=3,
                 ecolor='r', elinewidth=1, capthick=1, alpha=0.7, label='negative')

    plt.fill_between(steering_budgets, np.array(pos_means) - np.array(pos_stdevs),
                     np.array(pos_means) + np.array(pos_stdevs),
                     alpha=0.2, color='g')
    plt.fill_between(steering_budgets, np.array(neg_means) - np.array(neg_stdevs),
                     np.array(neg_means) + np.array(neg_stdevs),
                     alpha=0.2, color='r')

    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.xlabel('k')
    plt.xlim([1, budget_lim])
    plt.ylim([-1, 1])

    xlabels = ["1", "", "", "", "5", "10", "20", "50", "100"]
    plt.xticks(steering_budgets, xlabels)
    y_ticks = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
    plt.yticks(y_ticks)

    plt.title(dimension, fontsize=8)

    plt.grid(True, which='both', linestyle=':', linewidth=0.5)

    # Add legend
    plt.legend(loc='upper right', fontsize='small')

    plt.savefig(f"steerability_indices_{dimension}_{model_name}.png")
    plt.show()


def _extract_profiling_subset(profiling_datasets, model_id, dimension):
    dataframes = profiling_datasets[model_id][dimension]
    filtered_dataframes = [df[df['persona_dim'] == dimension] for df in dataframes]
    return filtered_dataframes


def process_profiles(profiles, profiling_datasets, belief_update, steering_budgets, save_dir):

    # todo: combine profiles across trials
    # profiles = _combine_profiles(profiles, method="")

    model_ids = list(profiles.keys())
    persona_dims = list(list(profiles.values()))[0].keys()

    for model_id in model_ids:
        filepath = os.path.join(save_dir, model_id)
        os.makedirs(filepath, exist_ok=True)

    # compute steerability indices
    base_profiles = {}
    steerability_indices = {}
    for model_id in model_ids:
        model_name = model_id.split("/")[-1]
        base_profiles[model_id] = {}
        steerability_indices[model_id] = {}
        for dimension in persona_dims:
            indices, params = compute_steerability_indices(profiles=profiles[model_id][dimension],
                                                           profiling_data=_extract_profiling_subset(profiling_datasets, model_id, dimension),
                                                           belief_update=belief_update,
                                                           steering_budgets=steering_budgets)
            steerability_indices[model_id][dimension] = indices
            base_profiles[model_id][dimension] = params

            # plot_indices(steerability_indices[model_id][dimension], steering_budgets, dimension, model_name)

    save_to_json(steerability_indices, os.path.join(filepath, "steerabiility_indices.json"))


def save_profile(profile, model_id, save_dir):
    file_path = os.path.join(save_dir, model_id)
    os.makedirs(file_path, exist_ok=True)
    filename = os.path.join(file_path, "beta_profiles.json")

    with open(filename, 'w') as f:
        json.dump(profile, f, indent=4)
