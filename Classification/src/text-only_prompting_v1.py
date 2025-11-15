#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2025-08-06
Author: Dan Schumacher

How to run:
    see ./Classification/bin/text-only_prompting.sh
"""

import json
from typing import Tuple
import os, sys, argparse, random
import numpy as np
from tqdm import tqdm
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.build_questions import (
    TASK_DESCRIPTION, TITLE_MAPPINGS, X_MAPPINGS, Y_MAPPINGS, LEGEND_MAPPINGS, EXTRA_INFO_MAPPINGS)
from global_utils.file_io import Split, load_train_test, append_jsonl
from global_utils.logging_utils import MasterLogger
from global_utils.image_prompter import Prompt, ImagePrompter


# ----------------------------------------------------
# Argument parsing
# ----------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Visual prompting using images + few-shot Prompt objects.")
    p.add_argument("--input_folder", type=str, required=True)
    p.add_argument("--normalize", type=int, choices=[0,1], default=0, help="If set, normalizes the time series before visualization.")
    p.add_argument("--model_name", type=str, default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=2) # TODO make default larger
    p.add_argument("--print_to_console", type=int, default=1)
    return p.parse_args()


def extract_letter_to_idx(model_output:str, mapping:dict) -> Tuple[str, int]:
    # Define the regex pattern to match content between the delimiters
    pattern = r"\[(.*?)\]"

    # Search for the pattern in the input string
    match = re.search(pattern, model_output)

    # Check if a match is found and extract the substring
    if match:
        letter = match.group(1)
    else:
        letter = "no letter found"

    # ensure that it is a letter
    result = mapping.get(letter, -1)
    return letter, result

# ----------------------------------------------------
# Setup: load dataset + logger
# ----------------------------------------------------
def set_up() -> Tuple[Split, Split, argparse.Namespace, MasterLogger]:
    args = parse_args()

    random.seed(42)
    np.random.seed(42)

    dataset = os.path.basename(os.path.normpath(args.input_folder))
    args.dataset = dataset

    # Logging
    logs_dir = "./Classification/logs/text-only_prompting/"
    os.makedirs(logs_dir, exist_ok=True)
    file_ext = "rpc" 
    log_path = os.path.join(logs_dir, f"{dataset}_{file_ext}.log")

    logger = MasterLogger(
        log_path=log_path,
        init=True,
        clear=True,
        print_to_console=args.print_to_console,
    )

    # ---- NEW: tags for normalization + TS presence ----
    norm_tag = "normalized" if args.normalize else "raw"
    args.norm_tag = norm_tag

    # Output location
    # optionally nest by norm/ts so runs don’t collide
    out_dir = f"./Classification/data/sample_generations/{dataset}/text-only_prompting/{norm_tag}/"
    os.makedirs(out_dir, exist_ok=True)

    args.out_file = os.path.join(out_dir, f"{file_ext}.jsonl")

    logger.info(f"Clearing output file {args.out_file}")
    with open(args.out_file, "w") as _:
        pass

    # Load train/test with artifacts
    train, test = load_train_test(
        args.input_folder,
        mmap=False,
        attach_artifacts=True,
        normalize=bool(args.normalize),
    )
    return train, test, args, logger

# ----------------------------------------------------
# Main execution
# ----------------------------------------------------
if __name__ == "__main__":

    train, test, args, logger = set_up()

    # Convert arrays
    idx_tr, X_tr, y_tr = np.asarray(train.idx), np.asarray(train.X), np.asarray(train.y).ravel()
    idx_te, X_te, y_te = np.asarray(test.idx), np.asarray(test.X), np.asarray(test.y).ravel()


    # Question used across dataset
    general_question = test.general_question or train.general_question
    general_question = general_question.strip()
    assert general_question, "General Question not available"

    prompter = ImagePrompter()
    prompter.model_name = args.model_name
    prompter.system_prompt = (
        TASK_DESCRIPTION[args.dataset.upper()] + \
        f' You will be given a multiple choice question and a time series.'
        f' Your job is to use the time series to answer the multiple choice question.'
        f' Think step by step and explain your reasoning. Then, provide a final answer.'
        f' The final answer must use the wording "The answer is [x]" where x is the answer to the multiple choice question.'
        f' Be sure to encapsulate x in square brackets.'
        f' \n\nHere is some additional information that may help you:\n'
        + EXTRA_INFO_MAPPINGS[args.dataset.upper()]
    )

    logger.info(f"Model = {args.model_name}")

    # ----------------------------------------------------
    # Process test set in batches
    # ----------------------------------------------------
    saved_example_prompt = False
    running_acc = []
    for start_of_batch in tqdm(range(0, len(test), args.batch_size), desc="Processing batches"):
        batch = test[start_of_batch : start_of_batch + args.batch_size]
        
        logger.info(f"Batch rows: {start_of_batch} → {start_of_batch+len(batch)}")

        # ------------------------------------------------
        # 1. Build few-shot examples for each test item
        # ------------------------------------------------
        fewshot_batches = []

        for row in batch:


            # ------------------------------------------------
            # RANDOM-PER-CLASS — number of shots is predetermined!
            example_indices = train.fixed_shot_idxs


            # ------------------------------------------------
            # Build Prompt objects for these example indices
            # ------------------------------------------------
            ex_prompts = []
            for ex_idx in example_indices:
                
                # PROMPT FOR AUOTO-COT
                TASK_DESCRIPTION[args.dataset.upper()]
                EXTRA_INFO_MAPPINGS[args.dataset.upper()]

                ex_X = X_tr[ex_idx]
                letter = train.label_maps["id_to_letter"][str(y_tr[ex_idx])]
                lbl_ex = f"The answer is [{letter}]"
                
                user_kwargs = {"question": general_question}

                user_kwargs["time_series"] = json.dumps(ex_X.tolist())

                prompt_kwargs = {
                    "user": user_kwargs,
                    "assistant": {"answer": lbl_ex},
                }

                ex_prompts.append(Prompt(**prompt_kwargs))

            fewshot_batches.append(ex_prompts)
        # ------------------------------------------------
        # 2. Build test queries (Prompt objects)
        # ------------------------------------------------
        queries = []
        meta_batch = []

        for row in batch:
            user_kwargs = {
                "question": general_question
            }

            user_kwargs["time_series"] = json.dumps(row.X.tolist())
            
            prompt_kwargs = {
                "user": user_kwargs,
            }
            meta_batch.append((row.idx, row.y))
            queries.append(Prompt(**prompt_kwargs))

        assert len(fewshot_batches) == len(queries), "Mismatch between few-shot sets and queries."
        
        # ------------------------------------------------
        # 3. Format prompts for each test query
        # ------------------------------------------------
        all_messages = []
        for examples, query in zip(fewshot_batches, queries):
            msgs = prompter.format_prompt(examples, query)
            all_messages.append(msgs)

        # SAVE EXAMPLE PROMPT IN MARKDOWN
        if not saved_example_prompt and len(all_messages) > 0:
            # Save ONE human-readable markdown of the very first prompt
            md_dir = os.path.join(os.path.dirname(args.out_file), "prompt_examples")
            os.makedirs(md_dir, exist_ok=True)
            file_ext = args.out_file.split("/")[-1].replace(".jsonl", ".md")
            md_path = os.path.join(md_dir, file_ext)

            prompter.export_prompt_markdown(
                examples=fewshot_batches[0],
                query=queries[0],
                out_md_path=md_path,
                save_images=False,
            )
            logger.info(f"Saved prompt markdown to: {md_path}")
            saved_example_prompt = True

        # ------------------------------------------------
        # 4. Model inference
        # ------------------------------------------------
        # logger.info(f"Sending {len(all_messages)} prompts to model...")
        results = prompter.get_completion(
            all_messages,
            temperature=args.temperature
            )

        if not isinstance(results, list):
            results = [results]

        # ------------------------------------------------
        # 5. Write results to JSONL
        # ------------------------------------------------
        for (idx_val, gt_val), result, messages in zip(meta_batch, results, all_messages):
            
            out_text = result.get("content", "").strip()
            letter, pred = extract_letter_to_idx(out_text, test.label_maps["letter_to_id"])
            correctness =np.asarray(gt_val) == pred
            running_acc.append(correctness)

            line = {
                "idx": np.asarray(idx_val).item(),
                "gt": np.asarray(gt_val).item(),
                "pred": pred, 
                "letter": letter,
                "model_output": out_text
            }

            append_jsonl(args.out_file, line)
        curr_acc = np.mean(running_acc)
        logger.info(f"Current accuracy at idx {np.asarray(idx_val).item()}:\t\t\t{curr_acc*100:.2f}%")

    logger.info(f"✅ Finished. Results saved → {args.out_file}")