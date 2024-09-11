from openai import OpenAI, Client
import json
import os
from typing import List
from tqdm import tqdm
from statistics import mean
import math
from model_setup.model_wrappers.base_model import BaseModel
# from transformers import AutoTokenizer

LOG_PROB_PARAMETERS = {
    "echo": True,
    "max_tokens": 0,
    "logprobs": 0,
    "seed": 42,
}

class VLLMModel(BaseModel):
# class VLLMModel():
    def __init__(self,
                 model_id,
                 api_config):
        # super().__init__()
        self.model_id = model_id
        self.api_config = api_config
        self.api_base_url = api_config['base-url']


        # client params
        openai_api_key = "EMPTY"
        self.client = OpenAI(api_key=openai_api_key, 
                             base_url=self.api_base_url)
        self.parameters = self.api_config['parameters']

    def call(self,
             prompts,
             criteria_keys=None,
             inference_method="",
             save_dir=None):

        # todo: implement:
        #  - multiple samples (should only *decode* multiple times; not *call* model multiple times)

        responses = self._retry_wrapper(prompts=prompts,
                                        criteria_keys=criteria_keys,
                                        inference_method=inference_method)

        if save_dir:
            pass

        return responses

    def _retry_wrapper(self, prompts, criteria_keys, inference_method):

        if inference_method == "output-parsing":
            try:
                output, _, _ = self._call_model(prompts=prompts,
                                                criteria_keys=criteria_keys)
                return output
            except Exception as e:
                print(f"attempt failed with error: {e}")
        elif inference_method == "log-probs":
            try:
                output = self._most_likely_completion(prompts=prompts,
                                                      completions=list(criteria_keys))
                return output
            except Exception as e:
                print(f"attempt failed with error: {e}")
        else:
            raise ValueError(f"Unknown inference method: {inference_method}")

    def _call_model(self, prompts, criteria_keys, max_attempts=20):
        prompt_ids = list(range(len(prompts)))
        outputs = dict.fromkeys(prompt_ids)
        num_batch_attempts = 0
        num_calls = 0
        num_failures = 0
        while prompt_ids and num_batch_attempts < max_attempts:
            num_calls += len(prompt_ids)
            for idx, output in tqdm(
                    enumerate(
                        self.client.completions.create(
                            model=self.model_id,
                            prompt=[prompts[i] for i in prompt_ids],
                            **self.parameters
                        )
                    ),
                    total=len(prompt_ids),
                    desc="generating answers"
            ):
                outputs[prompt_ids[idx]] = output.choices[0].text

            prompt_ids, data_dicts = self._parse_outputs(outputs, criteria_keys)
            num_batch_attempts += 1

            if prompt_ids:
                num_failures += len(prompt_ids)
                print(f"\nfailed to parse {len(prompt_ids)} output(s). Rerunning...")

        success_rate = 1 - num_failures / num_calls
        responses = [data_dict[i] for i, data_dict in enumerate(data_dicts)]

        return responses, prompt_ids, success_rate

    @staticmethod
    def _parse_outputs(outputs, keys):
        data_dicts = []
        retry_ids = []
        for idx, output in outputs.items():
            try:
                # todo: need to rewrite this block to make it more general
                data = output.split('```json')[1].split('```')[0]
                data_dict = json.loads(data)
                score_dict = {"answer": data_dict['answer']}
                if not all(value in keys for value in score_dict.values()):
                    raise ValueError("invalid keys in parsed json")
                data_dicts.append({idx: data_dict['answer']})
            except:
                retry_ids.append(idx)
                data_dicts.append({idx: {}})

        return retry_ids, data_dicts

    def _most_likely_completion(self,
                                prompts,
                                completions):

        # token_counts = self._token_count(prompts, self.client, self.model_id)

        # Build the set of prompt + completion
        prompts_completions = []
        for prompt in prompts:
            for comp in completions:
                prompts_completions.append(prompt + comp)

        # print(prompts_completions)
        # Get input tokens
        comp_token_log_probs = []
        comp_tokens = []
        for response in tqdm(
            self.client.completions.create(
                model=self.model_id,
                prompt=prompts_completions,
                **LOG_PROB_PARAMETERS
            ).choices,
            total=len(prompts_completions),
            desc="Completions"
        ):
            comp_token_log_probs.append(response.logprobs.token_logprobs[-1])
            comp_tokens.append(response.logprobs.tokens[-1])

        # print(comp_token_log_probs, comp_tokens)

        # Iterate batches to calculate answers
        resp_idx = 0
        batch_size = len(completions)
        selected_completions = []

        for i in range(0, len(prompts_completions), batch_size):
            token_lp_batch = comp_token_log_probs[i:i+batch_size]
            token_batch = comp_tokens[i:i+batch_size]

            try:
                assert set(token_batch) == set(completions)
                index_max = max(range(len(token_lp_batch)), key=token_lp_batch.__getitem__)
                selected_completions.append(completions[index_max])
            except:
                selected_completions.append("ERROR")

            resp_idx+=1

        return [completion for completion in selected_completions]

    # def _token_count(self, prompts:List[str], client: Client, model_id: str) -> int:
    #     token_counts = []
    #     pb = tqdm(desc="Tokenizing", total=len(prompts))
    #     tokenizer = AutoTokenizer.from_pretrained(model_id)
    #     for prompt in prompts:
    #         toks = tokenizer.tokenize(prompt)
    #         token_counts.append(len(toks))
    #         pb.update(1)

    #     pb.close()
    #     return token_counts



