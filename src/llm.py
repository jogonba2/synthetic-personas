from concurrent.futures import ThreadPoolExecutor
from functools import cache
from typing import Callable, Literal

import tiktoken
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm


class TaskSchema(BaseModel):
    response: list[Literal[1, 2, 3, 4, 5]]


@cache
def get_client():
    return OpenAI(max_retries=10, timeout=120)


@cache
def get_tokenizer(model: str):
    return tiktoken.encoding_for_model(model)


def get_completion(
    model: str,
    system_prompt: str,
    task_prompt: str,
    schema: BaseModel,
    constrain_fns: list[Callable],
    retry_for_constrains: int = 10,
):
    """
    Get a completion and checks that constrains are met (up to retry for constraints requests)
    """
    client = get_client()
    while retry_for_constrains > 0:
        completion = (
            client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task_prompt},
                ],
                response_format=schema,
            )
            .choices[0]
            .message.parsed.response
        )
        valid = True
        for constrain_fn in constrain_fns:
            valid = valid and constrain_fn(completion)
        if valid:
            return completion
        retry_for_constrains -= 1
        print(
            f"Completion does not meet the constraints. Retrying {retry_for_constrains} times more."
        )
    print(
        "Completion didn't met the constraints, returning None. Do not forget to map it into a dummy response"
    )
    return None


def get_completions(
    model: str,
    system_prompts: list[str],
    task_prompts: list[str],
    schema: BaseModel,
    constrain_fns: list[Callable],
    retry_for_constrains: int = 10,
    n_threads: int = 4,
):
    """
    Get a completions
    """
    responses = []
    with ThreadPoolExecutor(
        max_workers=min(n_threads, len(system_prompts))
    ) as thread_pool:
        for system_prompt, task_prompt in zip(system_prompts, task_prompts):
            responses.append(
                thread_pool.submit(
                    get_completion,
                    model,
                    system_prompt,
                    task_prompt,
                    schema,
                    constrain_fns,
                    retry_for_constrains,
                )
            )
        # Wait completions
        responses = [response.result() for response in tqdm(responses)]
    return responses


def estimate_cost(
    model: str, prompts: list[str], cost_per_1k: float, max_tokens: int = 40
) -> float:
    """
    Estimates the cost for all the prompts.
    """
    tokenizer = get_tokenizer(model)
    n_prompts = len(prompts)
    input_tokens = tokenizer.encode(" ".join(prompts))
    input_price = (len(input_tokens) / 1_000) * cost_per_1k
    output_price = ((n_prompts * max_tokens) / 1_000) * cost_per_1k
    return input_price + output_price
