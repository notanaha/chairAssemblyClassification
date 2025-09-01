#!/usr/bin/env python3
""" 
0_chairAssembly_01respApi.py is a standalone classifier for single-step analysis.
0_stepDetection.py is a parallel wrapper that calls chairAssemblyStep.run() multiple times
"""

import os, json
import time
import base64
import glob
import argparse
from typing import Literal, List

import openai
from pydantic import BaseModel, Field
from dotenv import load_dotenv


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class StepResponse(BaseModel):
    classification: Literal["yes", "no"] = Field(..., description="Classification of whether the images correspond to the step")
    confidence: int = Field(..., ge=1, le=5, description="Classification Score (1 as not belonging to the step, 5 as belonging to the step)")
    reasoning: str = Field(..., description="Explanation of the decision")

class StepResponseList(BaseModel):
    responses: List[StepResponse] = Field(..., description="List of responses for each test image")
    overall_result: Literal["yes", "no"] = Field(..., description="Overall classification of whether the images correspond to the step")
    reasoning: str = Field(..., description="Overall reasoning for the classification")

def main():
    parser = argparse.ArgumentParser(description="Run step classification for chair assembly images")
    parser.add_argument("--step_name", type=str, default="5_hexwrench_tight", help="Name of the assembly step")
    parser.add_argument(
        "--test_images",
        nargs='+',
        default=["frame_0063_t63.0s.jpg", "frame_0064_t64.0s.jpg", "frame_0065_t65.0s.jpg"],
        help="List of test image filenames"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    step_name = args.step_name
    test_images = args.test_images

    load_dotenv(override=True)
    aoai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    aoai_api_key = os.environ["AZURE_OPENAI_API_KEY"]
    api_version = os.environ["AZURE_OPENAI_API_VERSION"]

    client = openai.AzureOpenAI(
        azure_endpoint=aoai_endpoint,
        api_key=aoai_api_key,
        api_version=api_version
    )

    with open(os.path.join("./utils", f"{step_name}.txt"), "r", encoding="utf-8") as f:
        user_prompt_template = f.read()
    if args.verbose:
        print(f"Using system message: {step_name}.txt")

    messages = []
    content = []

    # Sample images
    content.append({"type": "input_text", 
                    "text": f"You are analyzing the productivity of a chair assembly process.The following images are examples of the chair assembly process."})
    sample_folder = f"./output/{step_name}/"
    image_files = sorted(glob.glob(os.path.join(sample_folder, "*.jpg")))
    for idx, image_path in enumerate(image_files, start=1):
        image_name = os.path.basename(image_path)
        base64_encoded = encode_image(image_path)
        if args.verbose:
            print(f"Encoding sample image {idx}: {image_name}")
        content.append({"type": "input_text", "text": f"### sample{idx}"})
        content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_encoded}", "detail": "high"})

    content.append({"type": "input_text", "text": user_prompt_template})


    # Test images
    test_folder = "./output/"
    for i, img_name in enumerate(test_images, start=1):
        base64_img = encode_image(os.path.join(test_folder, img_name))
        if args.verbose:
            print(f"Encoding test image {i}: {img_name}")
        content.append({"type": "input_text", "text": f"### test{i}"})
        content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_img}", "detail": "high"})

    content.append({"type": "input_text", "text": f"Classify each of the three images whether it belongs to step {step_name}. Return a JSON object with indent matching the given schema."})


    messages.append({"role": "user", "content": content})

    # Parse response
    if args.verbose:
        start_time = time.time()
        print(f"started parsing response")
    
    response = client.responses.parse(
        input=messages,
        #model="gpt-4.1", temperature=0, 
        model="gpt-5-mini", reasoning={"effort": "high", "summary": "auto"},
        text_format=StepResponseList
    )

    if args.verbose:
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

    print(response.output_text)

    data = json.loads(response.output_text)
    result = "yes" if any(label["classification"] == "yes" for label in data["responses"]) else "no"

    print("\nClassification result:", result)
    return result


if __name__ == "__main__":
    main()
