'''
2025-08-06
Author: Dan Schumacher
How to run:
   see ./Classification/bin/simple_prompting.sh
'''
import sys, os, json, argparse, random, re

from utils.sp_utils import extract_letter, to_text
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import numpy as np
from typing import List, Tuple, Union
from tqdm import tqdm

from global_utils.file_io import append_jsonl, load_train_test
from global_utils.get_prompter import get_prompter_class_and_kwargs
from global_utils.logging_utils import MasterLogger
from global_utils.prompter import GemmaPrompter, HFPrompter, OpenAIPrompter, Prompter

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Simple prompting using precomputed neighbors & maps from loader artifacts.")
    p.add_argument('--input_folder', type=str, required=True)
    p.add_argument('--prompt_path', type=str, required=True)
    p.add_argument('--model_type', type=str, default="gpt",
                   choices=["gpt", "llama", "mistral", "gemma"])
    p.add_argument('--n_shots', type=int, default=0, help="few-shot examples per input (capped at 10)")
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--batch_size', type=int, default=12)
    p.add_argument('--device_map', type=int, nargs='+', default=[0])
    p.add_argument('--show_prompt', type=int, default=0, choices=[0, 1])
    return p.parse_args()


# -------------------- Setup --------------------
def set_up() -> Tuple[argparse.Namespace, Union[GemmaPrompter, HFPrompter, OpenAIPrompter], MasterLogger, tuple]:
    args = parse_args()
    random.seed(42); np.random.seed(42)
    dataset = os.path.basename(os.path.normpath(args.input_folder))
    args.dataset = dataset
    logs_dir = "./Classification/logs/prompting"; os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{args.dataset}_NS{args.n_shots}.log")
    logger = MasterLogger(log_path=log_path, init=True, clear=True)

    out_dir = os.path.join("./Classification/data/generations", dataset, "simple_prompting")
    args.out_file = os.path.join(out_dir, f"NS{args.n_shots}.jsonl")
    os.makedirs(out_dir, exist_ok=True)

    # Load splits and attach artifacts (shot_idxs, label_maps, general_question)
    train, test = load_train_test(
        args.input_folder,
        attach_artifacts=True,
    )

    PrompterClass, shared_kwargs = get_prompter_class_and_kwargs(
        model_type=args.model_type, device_map=args.device_map, show_prompt=args.show_prompt,
    )

    prompter: Prompter = PrompterClass(
        prompt_path=args.prompt_path,
        prompt_headers={"question": "Question:", "timeseries": "Time Series:"},
        temperature=args.temperature,
        **shared_kwargs
    )

    logger.info("-"*60)
    logger.info(f"Running model: {args.model_type} on dataset: {args.dataset}")
    logger.info(f"[PATHS] input_folder={args.input_folder}")
    logger.info(f"[PATHS] output_path={args.out_file}")
    logger.info("-"*60)
    logger.info("TRAIN META DATA:")
    logger.info(train)
    logger.info("-"*60)
    logger.info("TEST META DATA:")
    logger.info(test)
    logger.info("-"*80)


    return train, test, args, prompter, logger

# -------------------- Main --------------------
if __name__ == "__main__":
    train, test, args, prompter, logger = set_up()

    # Arrays
    X_tr, y_tr = np.asarray(train.X), np.asarray(train.y).ravel()
    X_te, y_te = np.asarray(test.X),  np.asarray(test.y).ravel()

    # Shots cap
    n_shots = min(int(args.n_shots), 10, len(y_tr))
    logger.info(f"[INFO] Using {n_shots} few-shot examples per test query (cap=10).")

    # Artifacts
    topk = test.shot_idxs  # None or (N_test, 10)
    general_question = (test.general_question or "").strip()
    label_maps = train.label_maps or {}
    letter_to_id = label_maps.get("letter_to_id", {})

    # Batch rows
    test_rows = [
        {"idx": i, "ts": X_te[i], "label": int(y_te[i])}
        for i in range(len(y_te))
    ]

    # Process
    for start in tqdm(range(0, len(test_rows), args.batch_size), desc="Processing batches"):
        batch = test_rows[start : start + args.batch_size]
        batch_id = start // args.batch_size + 1
        logger.info(f"[BATCH] {batch_id} | size={len(batch)} | idx range [{start}:{start+len(batch)}]")

        # 1) Few-shots per input using Split helper
        per_input_examples = []
        for row in batch:
            i = int(row["idx"])
            neighbor_indices: List[int] = []
            if topk is not None and 0 <= i < topk.shape[0]:
                neighbor_indices = topk[i].tolist()[:n_shots]
            qas_shots = train.qas_from_indices(
                neighbor_indices,
                n_shots=n_shots,
                general_question=general_question
            )
            per_input_examples.append(qas_shots)

        # 2) Model inputs (+ meta)
        question_batch, batch_meta = [], []
        for row in batch:
            question_batch.append({
                "question": general_question,
                "timeseries": json.dumps(np.asarray(row["ts"]).tolist(), indent=2),
            })
            batch_meta.append((int(row["idx"]), int(row["label"])))

        # 3) Call model
        logger.info(f"[CALL] Sending {len(question_batch)} inputs to model…")
        results = prompter.get_completion(
            question_batch,
            parse=True,
            per_input_examples=per_input_examples,
        )
        if not isinstance(results, list):
            results = [results]

        # 4) Parse → LETTER → id → write JSONL
        for (idx_val, gt_val), result in zip(batch_meta, results):
            out_text = to_text(result).strip()
            letter = extract_letter(out_text)
            if not letter:
                logger.warning(f"[WARN] Could not parse letter for idx={idx_val}: {out_text}")
            pred_id = letter_to_id.get(letter, -1)

            line = {"idx": int(idx_val), "pred": int(pred_id), "gt": int(gt_val), "output": out_text}
            logger.info(f"[OUT] idx={idx_val} | gt={gt_val} | letter={letter or '??'} | pred={pred_id}")
            append_jsonl(args.out_file, line)

    logger.info(f"✅ All results saved to {args.out_file}")
