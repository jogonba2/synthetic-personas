"""
Task 2 -> redo the prompt to pass the whole set of statements instead of 1 by 1
"""

import json
from itertools import chain, product
from typing import Literal

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .llm import TaskSchema, estimate_cost, get_completions

ATTRIBUTES = [
    "gender",
    "life_stage",
    "educational_level",
    "income_level",
    "family_cultural_background",
    "personality_trait",
]

PERSONA_PROMPT_PLACEHOLDERS = [
    "gender",
    "gender_definition",
    "life_stage",
    "life_stage_definition",
    "educational_level",
    "educational_level_definition",
    "income_level",
    "income_level_definition",
    "family_cultural_background",
    "family_cultural_background_definition",
    "personality_trait",
    "personality_trait_definition",
]


def load_json(path):
    """
    Load a json file
    """
    with open(path, "r") as fr:
        return json.load(fr)


def format_prompt(prompt_template: str, features: dict) -> str:
    """
    Formats a prompt with named {} placeholders
    """
    return prompt_template.format(**features)


def get_persona_combinations(definitions) -> list[dict[str, str]]:
    """
    Get all the combinations of persona attributes.
    """
    genders = definitions["gender"].items()
    life_stage = definitions["life_stage"].items()
    educational_level = definitions["educational_level"].items()
    income_level = definitions["income_level"].items()
    family_background = definitions["family_cultural_background"].items()
    personality_trait = definitions["personality_trait"].items()
    return [
        {
            placeholder: attribute
            for placeholder, attribute in zip(
                PERSONA_PROMPT_PLACEHOLDERS, list(chain(*persona))
            )
        }
        for persona in list(
            product(
                genders,
                life_stage,
                educational_level,
                income_level,
                family_background,
                personality_trait,
            )
        )
    ]


def format_itemize(item_dict: dict) -> str:
    """
    Formats an itemize given a dictionary as {"1": "blabla", "2": ...}
    """
    return "\n".join(
        [f"{val}. {statement}" for val, statement in item_dict.items()]
    )


def get_prompts(
    json_config: dict,
    task: Literal["task_1", "task_2"],
    persona_combinations: list[dict[str, str]] = [],
    do_estimate_cost: bool = False,
    max_tokens_estimate: int = 40,
):
    # Get all the persona combinations and format the persona prompts
    if not persona_combinations:
        persona_combinations = get_persona_combinations(
            json_config["persona_definitions"]
        )

    persona_prompts = [
        format_prompt(json_config["persona_prompt_template"], persona)
        for persona in persona_combinations
    ]

    # Itemize the statements and format the task prompt
    statements = format_itemize(json_config[task]["statements"])
    task_prompt = format_prompt(
        json_config[task]["prompt"], {"statements": statements}
    )

    # Estimate costs
    if do_estimate_cost:
        all_prompts = [
            persona_prompt + "\n" + task_prompt
            for persona_prompt in persona_prompts
        ]
        print(
            estimate_cost(
                json_config["model"],
                all_prompts,
                json_config["cost_per_1k"],
                max_tokens=max_tokens_estimate,
            )
        )

    return persona_prompts, task_prompt


def get_label_encodings(json_config: dict):
    encoders = {}
    for attribute in ATTRIBUTES:
        encoders[attribute] = LabelEncoder().fit(
            list(json_config["persona_definitions"][attribute].keys())
        )
    return encoders


def encodings_to_df(label_encodings: dict):
    df_data = {"attribute": [], "class": [], "id": []}
    for attribute in label_encodings:
        classes = label_encodings[attribute].classes_.tolist()
        ids = (
            label_encodings[attribute]
            .transform(label_encodings[attribute].classes_)
            .tolist()
        )
        df_data["class"] += classes
        df_data["id"] += ids
        df_data["attribute"] += [attribute] * len(classes)
    return pd.DataFrame(df_data)


def sanitize_errors(
    completions, task: Literal["task_1", "task_2"], dummy_value: int = -1
):
    """
    Fills the errors with dummy values (error=case when the GPT completion does not meet the length requirement)
    """
    sanitized = []
    errors = 0
    for completion in completions:
        if completion is None:
            if task == "task_1":
                sanitized.append([dummy_value for _ in range(40)])
            else:
                sanitized.append([dummy_value for _ in range(10)])
            errors += 1
        else:
            sanitized.append(completion)

    if errors:
        print(
            f"[!] There were {errors} completions that do not meet the constraints and have been sanitized."
        )
    return sanitized


def run_task(
    json_config: dict,
    task: Literal["task_1", "task_2"],
    do_estimate_cost: bool = False,
    debug: bool = False,
    n_iters: int = 3,
):
    persona_combinations = get_persona_combinations(
        json_config["persona_definitions"]
    )

    persona_prompts, task_prompt = get_prompts(
        json_config,
        task,
        persona_combinations=persona_combinations,
        do_estimate_cost=do_estimate_cost,
        max_tokens_estimate=40 if task == "task_1" else 10,
    )

    if debug:
        persona_combinations = persona_combinations[:3]
        persona_prompts = persona_prompts[:3]

    # Get completions `n_iters` times
    completions = {}
    for i in range(n_iters):
        completions[i] = sanitize_errors(
            get_completions(
                model=json_config["model"],
                system_prompts=persona_prompts,
                task_prompts=[task_prompt for _ in range(len(persona_prompts))],
                schema=TaskSchema,
                constrain_fns=(
                    [lambda x: len(x) == 40]
                    if task == "task_1"
                    else [lambda x: len(x) == 10]
                ),
            ),
            task,
        )

    # Prepare results dataframe
    label_encodings = get_label_encodings(json_config)
    df_data = []
    results_column = "Need" if task == "task_1" else "Value"
    for persona_id, persona in enumerate(persona_combinations):
        if debug:
            print(
                persona_id,
                persona,
                persona_prompts[persona_id],
                task_prompt,
            )

        persona_data = {
            **{"#": persona_id, "iter": -1},
            **{
                key: label_encodings[key].transform([val])[0].item()
                for key, val in persona.items()
                if key in ATTRIBUTES
            },
        }
        for iter in range(n_iters):
            persona_data["iter"] = iter + 1
            row = {
                **persona_data,
                **{
                    f"{results_column} {idx+1}": answer
                    for idx, answer in enumerate(completions[iter][persona_id])
                },
            }
            df_data.append(row)
    df = pd.DataFrame(df_data)

    # Prepare label mapping dataframe
    mapping_df = encodings_to_df(label_encodings)

    # Store excel file
    with pd.ExcelWriter(f"results_{task}.xlsx", engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=task, index=False)
        mapping_df.to_excel(writer, sheet_name="Label Mapping", index=False)


if __name__ == "__main__":
    path = "etc/experiments/29-11-24/config.json"
    json_config = load_json(path)
    task = "task_1"
    run_task(json_config, task)
