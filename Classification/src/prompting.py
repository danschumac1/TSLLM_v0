#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2025-08-06
Author: Dan Schumacher

How to run:
    see ./Classification/bin/prompting.sh
"""

import json
from typing import Tuple
import os, sys, argparse, random
import numpy as np
from tqdm import tqdm
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.vis import plot_time_series
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
    p.add_argument("--use_vision", type=int, choices=[0,1], default=1, help="If set, includes the time series image in the prompt.")
    p.add_argument("--img_detail", type=str, choices=["auto","low", "high"], default="auto")
    p.add_argument("--visualization_method", type=str, choices=["NA", "line", "spectrogram"], default="line")
    p.add_argument("--normalize", type=int, choices=[0,1], default=0, help="If set, normalizes the time series before visualization.")
    p.add_argument("--include_ts", type=int, choices=[0,1], default=0, help="If set, includes the raw time series values in the prompt.")
    p.add_argument("--model_name", type=str, default="gpt-4o-mini")
    p.add_argument("--n_shots", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=2) # TODO make default larger
    p.add_argument("--print_to_console", type=int, default=1)
    p.add_argument("--clear_images", type=int, choices=[0,1], default=1, help="If set, clears existing images for the dataset.")
    p.add_argument("--shot_mode", type=str, choices=["random_per_class", "dtw"], default="random_per_class")
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

    if args.clear_images:
        img_dir = f"Classification/data/images/{dataset}"
        if os.path.exists(img_dir):
            import shutil

            print(f"Clearing image directory: {img_dir}")
            shutil.rmtree(img_dir)

    # Logging
    logs_dir = "./Classification/logs/prompting/"
    os.makedirs(logs_dir, exist_ok=True)
    if args.shot_mode == "random_per_class":
        log_path = os.path.join(logs_dir, f"{dataset}_rpc.log")
    else:
        log_path = os.path.join(logs_dir, f"{dataset}_{args.n_shots}-shot.log")

    logger = MasterLogger(
        log_path=log_path,
        init=True,
        clear=True,
        print_to_console=args.print_to_console,
    )

    # ---- NEW: tags for normalization + TS presence ----
    norm_tag = "norm" if args.normalize else "raw"
    ts_tag = "ts" if args.include_ts else "no-ts"
    args.norm_tag = norm_tag
    args.ts_tag = ts_tag

    # Output location
    type_str = "visual-prompting" if args.use_vision else "text-only-prompting"
    vis_method_str = args.visualization_method.replace(" ", "-")
    if args.visualization_method != "NA":
        type_str += f"/{vis_method_str}"

    # optionally nest by norm/ts so runs don’t collide
    out_dir = f"./Classification/data/sample_generations/{dataset}/{type_str}/{norm_tag}/{ts_tag}/"
    os.makedirs(out_dir, exist_ok=True)

    out_fname = (
        f"{args.n_shots}-shot_"
        f"{args.shot_mode}_"
        f"{'vision' if args.use_vision else 'text-only'}_"
        f"{args.visualization_method}_"
        f"{norm_tag}_{ts_tag}.jsonl"
    )
    args.out_file = os.path.join(out_dir, out_fname)

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

    logger.info(f"Loaded dataset: {train.dataset}")
    logger.info(f"Train size={len(train)}, Test size={len(test)}")
    logger.info(f"Output file: {args.out_file}")

    return train, test, args, logger

# ----------------------------------------------------
# Main execution
# ----------------------------------------------------
if __name__ == "__main__":

    train, test, args, logger = set_up()
    # Few shot examples location
    fewshot_dir = f"./Classification/data/images/examples/{args.dataset}"
    test_img_dir = f"./Classification/data/images/{args.dataset}/"
    os.makedirs(fewshot_dir, exist_ok=True)

    # Convert arrays
    idx_tr, X_tr, y_tr = np.asarray(train.idx), np.asarray(train.X), np.asarray(train.y).ravel()
    idx_te, X_te, y_te = np.asarray(test.idx), np.asarray(test.X), np.asarray(test.y).ravel()

    # Number of shots per example
    n_shots = min(args.n_shots, 10, len(train))
    logger.info(f"Using {n_shots} few-shot examples per test item.")

    # Question used across dataset
    general_question = test.general_question or train.general_question
    general_question = general_question.strip()
    assert general_question, "General Question not available"

    # Initialize our multimodal prompter

    if args.use_vision and args.include_ts:
        input_string = 'visualization and its corresponding values'
    elif args.use_vision:
        input_string = 'visualization'
    elif args.include_ts:
        input_string = 'time series values'
    else:
        raise ValueError("At least one of --use_vision or --include_ts must be set to 1.")

    

    prompter = ImagePrompter()
    prompter.model_name = args.model_name
    prompter.system_prompt = (
        TASK_DESCRIPTION[args.dataset.upper()] + \
        f' You will be given a multiple choice question and a time series {input_string}.'
        f' Your job is to use the {input_string} to answer the multiple choice question.'
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
            # DTW MODE — n_shots matters
            # ------------------------------------------------
            if args.shot_mode == "dtw":

                if n_shots == 0:
                    # Zero-shot DTW → no examples
                    example_indices = []
                else:
                    example_indices = []
                    if (
                        test.shot_idxs is not None 
                        and 0 <= row.idx < test.shot_idxs.shape[0]
                    ):
                        # These are indices into TRAIN
                        example_indices = (
                            test.shot_idxs[row.idx].tolist()[0][:n_shots]
                        )

            # ------------------------------------------------
            # RANDOM-PER-CLASS — number of shots is predetermined!
            # n_shots DOES NOT affect this mode.
            # ------------------------------------------------
            elif args.shot_mode == "random_per_class":
                example_indices = train.fixed_shot_idxs

            # ------------------------------------------------
            # Build Prompt objects for these example indices
            # ------------------------------------------------
            ex_prompts = []
            for ex_idx in example_indices:
                ex_X = X_tr[ex_idx]
                letter = train.label_maps["id_to_letter"][str(y_tr[ex_idx])]
                lbl_ex = f"The answer is [{letter}]"
                
                user_kwargs = {"question": general_question}

                if args.include_ts:
                    user_kwargs["time_series"] = json.dumps(ex_X.tolist())

                prompt_kwargs = {
                    "user": user_kwargs,
                    "assistant": {"answer": lbl_ex},
                }

                if args.use_vision:
                    prompt_kwargs["img_path"] = plot_time_series(
                        ex_X,
                        method=args.visualization_method,
                        title=TITLE_MAPPINGS[args.dataset.upper()],
                        xlabs=X_MAPPINGS[args.dataset.upper()],
                        ylabs=Y_MAPPINGS[args.dataset.upper()],
                        legends=LEGEND_MAPPINGS.get(args.dataset.upper(), None),
                        save_path=(
                            f"{fewshot_dir}/idx_{ex_idx}_"
                            f"{args.visualization_method[:5]}_{args.img_detail}.png"
                        ),
                        recreate=True,
                    )
                    prompt_kwargs["img_detail"] = args.img_detail

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

            if args.include_ts:
                user_kwargs["time_series"] = json.dumps(row.X.tolist())
            
            prompt_kwargs = {
                "user": user_kwargs,
            }
            if args.use_vision:
                prompt_kwargs["img_path"] = plot_time_series(
                    row.X, 
                    method=args.visualization_method,
                    title=TITLE_MAPPINGS[args.dataset.upper()],
                    xlabs=X_MAPPINGS[args.dataset.upper()],
                    ylabs=Y_MAPPINGS[args.dataset.upper()],
                    legends=LEGEND_MAPPINGS.get(f"{args.dataset.upper()}", None),
                    save_path=f"{test_img_dir}/idx_{row.idx}_{args.visualization_method[:5]}_{args.img_detail}.png",
                    recreate=True
                    )
                prompt_kwargs["img_detail"] = args.img_detail

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

            md_fname = (
                f"{args.n_shots}-shot_"
                f"{args.shot_mode}_"
                f"{'vision' if args.use_vision else 'text-only'}_"
                f"{args.visualization_method}_"
                f"{args.norm_tag}_{args.ts_tag}.md"
            )
            md_path = os.path.join(md_dir, md_fname)

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
