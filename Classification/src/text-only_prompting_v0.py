#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2025-08-06
Author: Dan Schumacher

How to run:
    python ./Classification/src/text-only_prompting_v0.py \
        --input_folder ./Classification/data/samples/har/ \
        --model_name gpt-4o-mini \
        --n_shots 3 \
        --batch_size 3 \
        --print_to_console 1
"""

import json
from typing import List, Optional, Tuple
import os, sys, argparse, random
import numpy as np
from tqdm import tqdm
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.vis import plot_time_series
from utils.build_questions import TASK_DESCRIPTION,  EXTRA_INFO_MAPPINGS, Y_MAPPINGS, LEGEND_MAPPINGS
from global_utils.file_io import Split, load_train_test, append_jsonl
from global_utils.logging_utils import MasterLogger
from global_utils.image_prompter import Prompt, ImagePrompter


# ----------------------------------------------------
# Argument parsing
# ----------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Text-only prompting using images + few-shot Prompt objects.")
    p.add_argument("--input_folder", type=str, required=True)
    p.add_argument("--model_name", type=str, default="gpt-4o-mini")
    p.add_argument("--n_shots", type=int, default=1)
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

def X_to_string(
        X:np.ndarray, y_val:str, legend_map: Optional[List[str]]=None) -> str:
    if legend_map:
        assert X.shape[1] > 1, "Xdim must be at least 2d (multivariate)"
        assert X.shape[1] == legend_map, "Xdim != to number of legends in map"
        new_dict = {}
        for Xdim, leg in zip(X, legend_map):
            new_dict[leg] = Xdim
        X_str_rep = f"{y_val.upper()}:\n{json.dumps(new_dict)}"
    else:
        X_str_rep = f"{y_val.upper()}:\n{json.dumps(X)}"
    return X_str_rep



# ----------------------------------------------------
# Setup: load dataset + logger
# ----------------------------------------------------
def set_up() -> Tuple[Split, Split, argparse.Namespace, MasterLogger]:
    random.seed(42)
    np.random.seed(42)

    args = parse_args()

    dataset = os.path.basename(os.path.normpath(args.input_folder))
    args.dataset = dataset

    # Logging
    logs_dir = "./logs/text-only_prompting/"
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{dataset}_NS{args.n_shots}.log")

    logger = MasterLogger(
        log_path=log_path,
        init=True,
        clear=True,
        print_to_console=args.print_to_console,
    )

    # Output location
    out_dir = f"./Classification/data/sample_generations/{dataset}/text-only_prompting/"
    os.makedirs(out_dir, exist_ok=True)
    args.out_file = os.path.join(out_dir, f"{args.n_shots}-shot.jsonl")
    logger.info(f"Clearing output file {args.out_file}")
    with open(args.out_file, "w") as _:
        pass


    # Load train/test with artifacts
    train, test = load_train_test(args.input_folder, attach_artifacts=True)

    logger.info(f"Loaded dataset: {train.dataset}")
    logger.info(f"Train size={len(train)}, Test size={len(test)}")
    logger.info(f"Output file: {args.out_file}")

    return train, test, args, logger




# ----------------------------------------------------
# Main execution
# ----------------------------------------------------
if __name__ == "__main__":

    train, test, args, logger = set_up()

    # Convert arrays
    idx_tr, X_tr, y_tr = np.asarray(train.idx), np.asarray(train.X), np.asarray(train.y).ravel()
    id2pos = {int(id_): i for i, id_ in enumerate(idx_tr)}

    idx_te, X_te, y_te = np.asarray(test.idx), np.asarray(test.X), np.asarray(test.y).ravel()

    # Number of shots per example
    n_shots = min(args.n_shots, 10, len(train))
    logger.info(f"Using {n_shots} few-shot examples per test item.")

    # Question used across dataset
    general_question = test.general_question or train.general_question
    general_question = general_question.strip()
    assert general_question, "General Question not available"

    # Initialize our multimodal prompter
    prompter = ImagePrompter()
    prompter.model_name = args.model_name
    prompter.system_prompt = (
        TASK_DESCRIPTION[args.dataset.upper()] + ' You will be given a multiple choice question and a time series.'
        ' Your job is to use the time series to answer the multiple choice question.'
        ' First, think step by step and explain your reasoning.'
        ' Then, to directly answer the question, use the wording "The answer is [x]" where x is the answer to the multiple choice question.'
        ' Be sure to encapsulate x in square brackets.'
        ' \n\nHere is some additional information that may help you:\n'
        + EXTRA_INFO_MAPPINGS[args.dataset.upper()]
    )

    logger.info(f"Model = {args.model_name}")

    # ----------------------------------------------------
    # Process test set in batches
    # ----------------------------------------------------
    saved_example_prompt = False
    for start_of_batch in tqdm(range(0, len(test), args.batch_size), desc="Processing batches"):
        batch = test[start_of_batch : start_of_batch + args.batch_size]
        logger.info(f"Batch rows: {start_of_batch} → {start_of_batch+len(batch)}")

        # ------------------------------------------------
        # 1. Build few-shot examples for each test item
        # ------------------------------------------------
        fewshot_batches = []  # list of List[Prompt]

        for row in batch:

            # Get neighbor indices for few-shots
            neighbor_indices = []
            if test.shot_idxs is not None and 0 <= row.idx < test.shot_idxs.shape[0]:
                # these are indices into the TRAIN set
                neighbor_indices = test.shot_idxs[row.idx].tolist()[0][:n_shots]

            # Build Prompt objects for each few-shot neighbor
            ex_prompts = []
            for ni in neighbor_indices:
                ts_ex = X_tr[ni]
                letter = train.label_maps["id_to_letter"][json.dumps(y_tr[ni])]
                lbl_ex = f"The answer is [{letter}]" # TODO this would be nice to automate eventually.

                # Build few-shot Prompt (with correct answer)
                ex_prompts.append(
                    Prompt(
                        user={
                            "question": general_question,
                            "time_series": json.dumps(list(ts_ex))},
                        assistant={"answer": str(lbl_ex)},
                    )
                )

            fewshot_batches.append(ex_prompts)

        # ------------------------------------------------
        # 2. Build test queries (Prompt objects)
        # ------------------------------------------------
        queries = []
        meta_batch = []

        for row in batch:
            test_idx = row.idx

            queries.append(
                Prompt(
                    user={
                        "question": general_question,
                        "time_series": json.dumps(list(row.X))},
                )
            )
            meta_batch.append((test_idx, row.y))

        # ------------------------------------------------
        # 3. Format prompts for each test query
        # ------------------------------------------------
        all_messages = []
        for examples, query in zip(fewshot_batches, queries):
            msgs = prompter.format_prompt(examples, query)
            all_messages.append(msgs)

        # ------------------------------------------------
        # 4. Model inference
        # ------------------------------------------------

        if not saved_example_prompt and len(all_messages) > 0:
            # Save ONE human-readable markdown of the very first prompt
            md_dir = f"./Classification/data/sample_generations/{args.dataset}/text-only/prompt_examples"
            os.makedirs(md_dir, exist_ok=True)
            md_path = os.path.join(md_dir, f"{args.n_shots}-shot_prompt.md")

            # Save images next to the markdown in a subfolder called "images"
            prompter.export_prompt_markdown(
                examples=fewshot_batches[0],
                query=queries[0] ,
                out_md_path=md_path,
                save_images=False,
            )
            logger.info(f"Saved prompt markdown to: {md_path}")
            saved_example_prompt = True

        logger.info(f"Sending {len(all_messages)} prompts to model...")
        results = prompter.get_completion(
            all_messages,
            temperature=0.0
            )

        if not isinstance(results, list):
            results = [results]

        # ------------------------------------------------
        # 5. Write results to JSONL
        # ------------------------------------------------
        for (idx_val, gt_val), result, messages in zip(meta_batch, results, all_messages):
            
            out_text = result.get("content", "").strip()
            letter, pred = extract_letter_to_idx(out_text, test.label_maps["letter_to_id"])
            line = {
                "idx": np.asarray(idx_val).item(),
                "gt": np.asarray(gt_val).item(),
                "pred": pred, 
                "letter": letter,
                "model_output": out_text
            }

            append_jsonl(args.out_file, line)
            logger.info(f"Line {np.asarray(idx_val).item()}: {pred == np.asarray(gt_val).item()}\t | {out_text}")
    logger.info(f"✅ Finished. Results saved → {args.out_file}")
