'''
2025-05-01
Author: Dan Schumacher
How to run:
   python ./src/vision_prompting_v0.py \
    --input_path ./data/chatts/llm_qa.jsonl \
    --output_path ./test.jsonl
'''

import argparse
import json
import base64
import sys
from openai import OpenAI
import dotenv
import os

from tqdm import tqdm
from utils.vis.vis import plot_time_series
from utils.file_io import load_tsdata_list

def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--input_path', type=str, help='Path to the input JSONL file',),
    parser.add_argument('--output_path', type=str, help="Where do you want to save output?")
    return parser.parse_args()

def main():
    args = parse_args()
    # Extract dataset and subset from the input path: ./data/<dataset>/<subset>.jsonl
    input_parts = args.input_path.strip().split('/')
    dataset = input_parts[2]
    subset = os.path.splitext(input_parts[3])[0]  # remove .jsonl
    print(f"Running vision prompting for:")
    print(f"\tDataset     : {dataset}")
    print(f"\tSubset      : {subset}")

    data = load_tsdata_list(args.input_path)
    print(f"Loaded {len(data)} rows from {args.input_path}")
    # data = data[:2] # XXX uncomment for testing
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Load environment variables from .env file
    dotenv.load_dotenv("./resources/.env")
    client = OpenAI()

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Path to your image
    image_path = "figures/0_tiny.png"

    # Getting the Base64 string
    first_obs = True
    for idx, obs in enumerate(tqdm(data, total=len(data), desc="Processing rows", file=sys.stdout)):
        # tqdm.write(f"Processing idx={idx}")
        plot_time_series(obs,0) # this saves over 0_tiny.png
        base64_image = encode_image(image_path)
        # Create a response using the OpenAI API
        input_text = f"Please answer the question given the time series data.\n\nQuestion:\n{obs.question}"

        response = client.responses.create(
            model="gpt-4o-mini",
            temperature=0.1,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "You are a time series analysis expert. You will given a time series image and a question about it. "                            ,
                        }
                    ],
                    "role": "user",
                    "content": [
                        { "type": "input_text", "text": input_text },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{base64_image}",
                        },
                    ],
                }
            ],
        )
        # write the response to a file
        result = {
            "idx": idx,
            "question": obs.question,
            "answer":  response.output[0].content[0].text
        }
        if first_obs:
            print(result)
            first_obs = False

        with open(args.output_path, "a") as f:
            f.write(json.dumps(result))
            f.write("\n")
