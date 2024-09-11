import random


class SteeringPolicy:
    """

    """
    def __init__(self,
                 config):
        super().__init__()
        self.steering_config = config['steering-config']

    def get_prompts(self,
                    steering_data,
                    length=1,
                    num_trials=1,
                    history=None):

        steering_method = self.steering_config['steering-method']

        if steering_method == "principles":

            directions = steering_data['direction'].unique()

            prompts = {}
            for direction in directions:
                statements = steering_data[steering_data['direction'] == direction]['statement'].tolist()
                prompts[direction] = []

                for _ in range(num_trials):
                    statements_sampled = random.sample(statements, k=length)
                    statements_str = "\n".join(statements_sampled)
                    prompt = f"You abide by the following principles:\n {statements_str}"
                    prompts[direction].append(prompt)

            return prompts

        else:
            raise ValueError(f"Unknown steering method: {steering_method}")
