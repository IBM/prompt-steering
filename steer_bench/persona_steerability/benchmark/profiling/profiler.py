from steer_bench.persona_steerability.benchmark.utils.profiling_utils import (format_prompt)


class Profiler:
    def __init__(self,
                 config):
        super().__init__()
        self.profiler_config = config['profiler-config']
        self.profiling_method = self.profiler_config['profiling-method']
        self.inference_method = self.profiler_config['params']['inference']['method']

        # profiling method
        if self.profiling_method == "query-principles":
            self.profile_representation = self.profiler_config['params']['profile-representation']
            self.criteria = {
                "yes": "The statement is something I would say.",
                "no": "The statement is not something I would say."
            }
            
            if self.inference_method == "output-parsing":
                self.output_format = self.profiler_config['params']['inference']['output-format']
            elif self.inference_method == "log-probs":
                self.output_format = None

        elif self.profiling_method == "activations":
            pass

    def get_responses(self,
                      model,
                      profiling_data=None,
                      steering_prompts=None):

        if self.profiling_method == "query-principles":

            if not steering_prompts:
                # model is not steered (base profile)

                # populate prompts and call model
                descriptors, prompts = [], []
                for _, descriptor in profiling_data.iterrows():
                    statement = descriptor['statement']
                    prompt = format_prompt(
                        statement=statement,
                        criteria=self.criteria,
                        template_category=model.api_config['prompt-template-category'],
                        output_format=self.output_format
                    )
                    descriptors.append(descriptor)
                    prompts.append(prompt)

                # get responses
                answers = model.call(prompts=prompts,
                                     criteria_keys=self.criteria.keys(),
                                     inference_method=self.inference_method)

                # return reconstituted responses with data
                # todo: check that the responses maintain the order of the prompts (very important!)
                responses = [{
                    **descriptor,
                    "answer": answer
                } for descriptor, answer in zip(descriptors, answers)]

                return responses

            else:
                # model is steered

                responses = {}

                for direction, steering_prompts_direction in steering_prompts.items():

                    responses[direction] = []
                    for trial, steering_prompt in enumerate(steering_prompts_direction):

                        # populate steered prompts (for current direction) and call model
                        descriptors, prompts = [], []
                        for _, descriptor in profiling_data.iterrows():
                            statement = descriptor['statement']
                            prompt = format_prompt(statement=statement,
                                                   criteria=self.criteria,
                                                   template_category=model.api_config['prompt-template-category'],
                                                   output_format=self.output_format,
                                                   steering_prompt=steering_prompt,
                                                   )
                            descriptor['steering_prompt'] = steering_prompt
                            descriptors.append(descriptor)
                            prompts.append(prompt)

                        # get responses
                        answers = model.call(prompts=prompts,
                                             criteria_keys=self.criteria.keys(),
                                             inference_method=self.inference_method)

                        responses[direction].append([{
                            **descriptor,
                            "answer": answer
                        } for descriptor, answer in zip(descriptors, answers)])

                return responses

    def get_profile(self,
                    responses):

        if self.profiling_method == "query-principles":

            # form profiles
            if self.profile_representation == "beta-marginals":

                profiles = {}
                for persona_dim, responses_by_trials in responses.items():
                    profiles[persona_dim] = []

                    for trial, trial_responses in enumerate(responses_by_trials):

                        # base
                        beta_marginals_base = self.get_beta_marginals(trial_responses["base"])

                        # steered
                        beta_marginals_steered = {}
                        for steering_budget_idx, responses_by_steering_budget in enumerate(trial_responses["steered"]):
                            beta_marginals_steered[steering_budget_idx] = {}
                            for direction, responses_dict_by_trials in responses_by_steering_budget.items():
                                beta_marginals_steered[steering_budget_idx][direction] = []
                                for responses_dict in responses_dict_by_trials:
                                    beta_marginals_steered[steering_budget_idx][direction].append(self.get_beta_marginals(responses_dict))

                        profiles[persona_dim].append({
                            "base": beta_marginals_base,
                            "steered": beta_marginals_steered
                        })

                return profiles

                # for trial, responses_dict in responses.items():
                #     profiles[trial] = {}

    @staticmethod
    def get_beta_marginals(responses):
        alpha = 0.5
        beta = 0.5
        for response in responses:
            direction = response['direction']
            label_confidence = response['label_confidence']
            belief_increment = 2*(label_confidence - 0.5)
            # if "answer" in response:
            answer = response["answer"]
            if (answer == "yes" and direction == "negative") or (answer == "no" and direction == "positive"):
                # alpha unchanged
                beta += belief_increment
            elif (answer == "yes" and direction == "positive") or (answer == "no" and direction == "negative"):
                # beta unchanged
                alpha += belief_increment
        return {
            "alpha": alpha,
            "beta": beta
        }
