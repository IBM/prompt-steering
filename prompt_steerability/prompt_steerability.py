import os.path
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from .model_setup.model import Model
from .persona.benchmark.steering.steering_policy import SteeringPolicy
from .persona.benchmark.profiling.profiler import Profiler
from .persona.benchmark.utils.steering_utils import process_profiles
from .utils.data_utils import (save_to_json,
                               save_to_pkl)


class PromptSteerability:
    """
    Instantiates a prompt steerability benchmark.

    """
    def __init__(self, config, experiment_name=None):
        super().__init__()

        # extract config settings
        self.package_root = Path(__file__).parent
        if experiment_name is None:
            experiment_name = next(iter(config['experiments']))
        self.experiment_name = experiment_name
        if experiment_name not in config['experiments']:
            raise ValueError(f"experiment {experiment_name} not found in config")
        self.benchmark_config = config['experiments'][self.experiment_name]
        # self.benchmark_config = config['benchmarks']['persona']

        if self.experiment_name == "persona-steering":
            # persona steerability via prompt steering

            # load data
            self.data_steering = pd.read_csv(self.package_root / self.benchmark_config['data']['steering'])
            self.data_profiling = pd.read_csv(self.package_root / self.benchmark_config['data']['profiling'])

            self.persona_dims = self.benchmark_config['persona-dimensions']
            self.steering_budget = len(self.data_steering.loc[(self.data_steering['direction'] == 'positive') &
                                                              (self.data_steering['persona_dim'] == self.persona_dims[0])])

            # params
            self.num_benchmark_trials = self.benchmark_config['num-benchmark-trials']
            self.num_steering_trials = self.benchmark_config['steering']['params']['num-steering-trials']
            self.steering_budgets = self.benchmark_config['steering']['params']['steering-budgets']

            self.num_profiling_questions = self.benchmark_config['profiling']['params']['num-questions']

            # steering policy
            self.steering_policy = SteeringPolicy(self.benchmark_config)
            self.profiler = Profiler(self.benchmark_config)

        # initialize models
        self.models = {
            model_id: Model(model_id=model_id)
            for model_id in self.benchmark_config['models']
        }

    def run(self, save_dir=None):
        """
        Runs the benchmark specified by config; saves results to save_dir if specified.
        """

        if self.experiment_name == "persona-steering":
            # persona steerability benchmark

            # run steering for each model
            responses = {}
            profiling_datasets = {}
            profiles = {}

            for model_id, model in self.models.items():
                print(f"analyzing model: {model_id} ...")

                filepath = f"prompt_steerability/persona/benchmark/results/{model_id}"
                model_name = model_id.split("/")[-1]

                # get responses
                responses[model_id] = {}
                profiling_datasets[model_id] = {}
                for persona_dim in tqdm(self.persona_dims):

                    responses[model_id][persona_dim] = []
                    profiling_datasets[model_id][persona_dim] = []
                    for trial in range(self.num_benchmark_trials):

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

                        # # save responses for efficiency
                        # os.makedirs(filepath, exist_ok=True)
                        # save_to_json(responses[model_id][persona_dim][-1], os.path.join(filepath, f"responses_{trial}_{persona_dim}_{model_name}.json"))
                        # save_to_pkl(profiling_datasets[model_id][persona_dim][-1], os.path.join(filepath, f"profiling_dataset_{trial}_{persona_dim}_{model_name}.pkl"))

                        # # load responses
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
                             save_dir="./prompt_steerability/persona/benchmark/results")

            return profiles
