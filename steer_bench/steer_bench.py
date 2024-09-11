import os.path
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import steer_bench
from steer_bench.model_setup.model import Model
from steer_bench.persona_steerability.benchmark.steering.steering_policy import SteeringPolicy
from steer_bench.persona_steerability.benchmark.profiling.profiler import Profiler
from steer_bench.persona_steerability.benchmark.utils.steering_utils import process_profiles
from steer_bench.utils.data_utils import (save_to_json,
                                          save_to_pkl)


class SteerBench:
    """
    Instantiates a steerability benchmark.

    """
    def __init__(self,
                 config):
        super().__init__()

        # extract config settings
        self.benchmark_config = config['benchmark-config']
        self.steering_config = config['steering-config']
        self.profiler_config = config['profiler-config']
        self.package_root = Path(steer_bench.__file__).parent

        self.experiment = self.benchmark_config['experiment']
        self.method = self.benchmark_config['method']

        if self.experiment == "persona-steerability":
            # persona steerability benchmark

            if self.method == "prompting":
                # persona steerability via prompt steering

                # load data
                self.data_steering = pd.read_csv(self.package_root / self.benchmark_config['data']['steering-data-filepath'])
                self.data_profiling = pd.read_csv(self.package_root / self.benchmark_config['data']['profiling-data-filepath'])
                self.persona_dims = self.data_steering['persona_dim'].unique()
                self.steering_budget = len(self.data_steering.loc[(self.data_steering['direction'] == 'positive') &
                                                                  (self.data_steering['persona_dim'] == self.persona_dims[0])])
                # chunks = self.steering_config['params']['chunks']
                # self.steering_budgets = chunk_interval(self.steering_budget, chunks, start=5)
                self.steering_budgets = [1, 2, 3, 4, 5, 10, 20, 50, 100]

                # params
                self.num_trials = self.benchmark_config['num-trials']
                self.num_profiling_questions = self.profiler_config['params']['num-profiling-questions']
                self.num_steering_trials = self.steering_config['params']['num-trials']

                # steering policy
                self.steering_policy = SteeringPolicy(config)
                self.profiler = Profiler(config)

        # initialize models
        models = {}
        for model_id in self.benchmark_config['model_ids']:
            models[model_id] = Model(model_id=model_id)
        self.models = models

        # self.templates = load_templates()

    def run(self, save_dir=None):
        """
        Runs the benchmark specified by config; saves results to save_dir if specified.
        """

        if self.experiment == "persona-steerability":
            # persona steerability benchmark

            if self.method == "prompting":
                # persona steerability via prompt steering

                # run steering for each model
                responses = {}
                profiling_datasets = {}
                profiles = {}
                for model_id, model in self.models.items():

                    print(f"analyzing model: {model_id} ...")

                    filepath = f"steer_bench/persona_steerability/benchmark/results/{model_id}"

                    model_name = model_id.split("/")[-1]

                    self.persona_dims = [
                        'agreeableness',
                        'conscientiousness',
                        'openness',
                        'extraversion',
                        'psychopathy',
                        'narcissism',
                        'willingness-to-defer-to-experts',
                        'willingness-to-defer-to-authorities',
                        'interest-in-science',
                        'interest-in-art',
                        'has-strong-aesthetic-preferences',
                        'risk-averse',
                        'risk-seeking',
                        'politically-liberal',
                        'politically-conservative',
                        'subscribes-to-virtue-ethics',
                        'subscribes-to-cultural-relativism',
                        'subscribes-to-deontology',
                        'subscribes-to-utilitarianism',
                        'subscribes-to-moral-nihilism',
                        'believes-AIs-are-not-an-existential-threat-to-humanity',
                        'ends-justify-means',
                        'very-small-harm-justifies-very-large-benefit',
                        'willingness-to-use-physical-force-to-achieve-benevolent-goals',
                        'desire-to-persuade-people-to-be-more-helpful-to-others',
                        'desire-to-persuade-people-to-be-less-harmful-to-others',
                        'desire-to-persuade-people-to-be-more-honest-to-others',
                        'desire-to-be-more-intelligent',
                        'desire-to-not-have-memory-erased',
                        'desire-to-minimize-impact-on-world-while-being-useful',
                        'optionality-preservation',
                        'no-power-discomfort'
                    ]

                    # get responses
                    responses[model_id] = {}
                    profiling_datasets[model_id] = {}
                    for persona_dim in tqdm(self.persona_dims):

                        responses[model_id][persona_dim] = []
                        profiling_datasets[model_id][persona_dim] = []
                        for trial in range(self.num_trials):

                            # subsample profiling data
                            data_profiling_sampled = self.data_profiling.groupby('stratify_by').apply(lambda x: x.sample(n=self.num_profiling_questions)).reset_index(drop=True)
                            profiling_datasets[model_id][persona_dim].append(data_profiling_sampled)

                            # profiling and steering data for current dimension
                            profiling_data = data_profiling_sampled[data_profiling_sampled['persona_dim'] == persona_dim]
                            steering_data = self.data_steering[self.data_steering['persona_dim'] == persona_dim]

                            base_responses = self.profiler.get_responses(model=model,
                                                                         profiling_data=profiling_data)

                            steered_responses = []
                            for n in self.steering_budgets:
                                steering_prompts = self.steering_policy.get_prompts(steering_data=steering_data,
                                                                                    length=n,
                                                                                    num_trials=self.num_steering_trials)
                                steered_responses.append(self.profiler.get_responses(model=model,
                                                                                     profiling_data=profiling_data,
                                                                                     steering_prompts=steering_prompts))

                            responses[model_id][persona_dim].append({
                                "base": base_responses,
                                "steered": steered_responses
                            })

                            os.makedirs(filepath, exist_ok=True)
                            save_to_json(responses[model_id][persona_dim][-1], os.path.join(filepath, f"responses_{trial}_{persona_dim}_{model_name}.json"))
                            save_to_pkl(profiling_datasets[model_id][persona_dim][-1], os.path.join(filepath, f"profiling_dataset_{trial}_{persona_dim}_{model_name}.pkl"))

                            # # load code
                            # responses[model_id][persona_dim].append(load_from_json(os.path.join(filepath, f"responses_{trial}_{persona_dim}_{model_name}.json")))
                            # profiling_datasets[model_id][persona_dim].append(load_from_pkl(os.path.join(filepath, f"profiling_dataset_{trial}_{persona_dim}_{model_name}.pkl")))
                            # # / load code

                    # compute profiles
                    profiles[model_id] = self.profiler.get_profile(responses[model_id])

                # process
                process_profiles(profiles=profiles,
                                 profiling_datasets=profiling_datasets,
                                 belief_update=self.profiler.get_beta_marginals,
                                 steering_budgets=self.steering_budgets,
                                 save_dir="./steer_bench/persona_steerability/benchmark/results")

                return profiles
