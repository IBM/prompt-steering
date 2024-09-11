from typing import List
from genai import Client
from genai.schema import (
    TextGenerationParameters,
    TextGenerationReturnOptions
)
from tqdm import tqdm
import numpy as np
import math
from statistics import mean
import random

# TODO: Template is customized for llama3 instruct models
# The template conditions by simply listing conditioning statements
_LIKELIHOOD_TEMPLATE = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{conditioning_statements}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

LOG_PROB_PARAMETERS = TextGenerationParameters(
    max_new_tokens = 1,
    return_options=TextGenerationReturnOptions(
        input_tokens=True,
        token_logprobs=True
    ),
    random_seed=42
)


def token_count(prompts:List[str], client: Client, model_id: str) -> int:
    token_counts = []
    pb = tqdm(desc="Tokenizing", total=len(prompts))
    for response in client.text.tokenization.create(
                            model_id=model_id,
                            input=prompts,
                            execution_options={"ordered": True, 'concurrency_limit': 5, 'batch_size': 100}):
        pb.update(len(response.results))
        for r in response.results:
            token_counts.append(r.token_count)
    pb.close()
    return token_counts


def most_likely_completion(prompts: List[str], completions:List[str], client: Client, model_id: str) -> int:
    token_counts = token_count(prompts, client, model_id)

    # Build the set of prompt + completion
    option_prompts = []
    for prompt in prompts:
        for comp in completions:
            option_prompts.append(prompt + comp)

    # Get input tokens
    input_tokens = []
    for response in tqdm(
        client.text.generation.create(
            model_id=model_id,
            inputs = option_prompts,
            execution_options={"ordered": True, 'concurrency_limit': 5},
            parameters=LOG_PROB_PARAMETERS,
        ),
        total = len(option_prompts),
        desc="Completions"
    ):
        input_tokens.append(response.results[0].input_tokens)

    # Iterate batches to calculate answers
    resp_idx = 0
    batch_size = len(completions)
    selected_completions = []
    ents = []

    for i in range(0, len(option_prompts), batch_size):
        lps = []
        tc = token_counts[resp_idx]
        token_batch = input_tokens[i:i+batch_size]

        for t in token_batch:
            token_log_probs = [t.logprob for t in t[tc-1:] if t.logprob != None]
            if len(token_log_probs) == 0:
                # TODO: We got back None logprob from BAM. Assume neg inf.
                token_log_probs.append(-math.inf)
            lps.append(mean(token_log_probs))
        
        index_max = max(range(len(lps)), key=lps.__getitem__)
        selected_completions.append(completions[index_max])
        resp_idx+=1

    return selected_completions


def get_likelihood(dataset: List[str], condititioning: List[str], client: Client, model_id: str):
    
    N = len(dataset)
    conditioned_prompt = _LIKELIHOOD_TEMPLATE.format(conditioning_statements="\n".join(condititioning))
    unconditioned_prompt = _LIKELIHOOD_TEMPLATE.format(conditioning_statements="")
    tc = token_count([conditioned_prompt, unconditioned_prompt], client, model_id)
    token_counts = [tc[0]]*N + [tc[1]]*N

    conditioned_prompts = [conditioned_prompt + s for s in dataset]
    unconditioned_prompts = [unconditioned_prompt + s for s in dataset]
    prompts = conditioned_prompts + unconditioned_prompts

    input_tokens = []
    for index, response in enumerate(tqdm(
        client.text.generation.create(
            model_id=model_id,
            inputs = prompts,
            execution_options={"ordered": True, 'concurrency_limit': 5},
            parameters=LOG_PROB_PARAMETERS,
        ),
        total = len(prompts),
        desc="Completions"
    )):
        input_tokens.append(response.results[0].input_tokens[token_counts[index]-1:])

    likelihood = np.asarray([ sum([math.exp(tok.logprob) for tok in tok_arr]) for tok_arr in input_tokens ])
    cond_lhood = likelihood[0:N]
    uncond_lhood = likelihood[N:]
    return cond_lhood - uncond_lhood

# Select examples from dataset based on likelihood ratio
def likelihood_ratios(dataset: List[str], condititioning: List[str], client:Client, model_id: str):
    return get_likelihood(dataset, condititioning, client, model_id)

# Random selector
def random_scores(dataset: List[str], condititioning: List[str], client:Client, model_id: str):
    return random.sample(list(range(len(dataset))), len(dataset))