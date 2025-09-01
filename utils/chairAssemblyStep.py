# chairAssemblyStep.py
#!/usr/bin/env python3
import os, json, time, base64, glob, argparse
from typing import Literal, List
from pathlib import Path

import openai
from pydantic import BaseModel, Field
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
PROMPT_DIR = BASE_DIR
OUTPUT_DIR = ROOT_DIR / "output" 

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


def run(step_name: str, test_images: List[str], verbose: bool = True) -> str:
    load_dotenv(override=True)
    aoai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    aoai_api_key  = os.environ["AZURE_OPENAI_API_KEY"]
    api_version   = os.environ["AZURE_OPENAI_API_VERSION"]

    client = openai.AzureOpenAI(
        azure_endpoint=aoai_endpoint,
        api_key=aoai_api_key,
        api_version=api_version
    )

    # プロンプトテンプレート
    with open(PROMPT_DIR / f"{step_name}.txt", "r", encoding="utf-8") as f:
        user_prompt_template = f.read()
    if verbose:
        print(f"Using system message: {step_name}.txt")

    messages = []
    content = []

    # Prompt part 1/3
    content.append({"type": "input_text",
                    "text": "You are analyzing the productivity of a chair assembly process."})

    # Attach Sample Images
    sample_folder = OUTPUT_DIR / step_name
    image_files = sorted(glob.glob(str(sample_folder / "*.jpg"))) 
    for idx, image_path in enumerate(image_files, start=1):
        image_name = os.path.basename(image_path)
        base64_encoded = encode_image(image_path)
        if verbose:
            print(f"Encoding sample image {idx}: {image_name}")
        content.append({"type": "input_text", "text": f"### sample{idx}"})
        content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_encoded}", "detail": "high"})

    # Prompt part 2/3
    content.append({"type": "input_text", "text": user_prompt_template})

    content.append({"type": "input_text",
                    "text": "\n## Now, you will see three test images, each representing a one-second-apart frame from the same video. \nAnswer the question that follows.\n"})

    # Attach Test Images
    test_folder = OUTPUT_DIR
    for i, img_name in enumerate(test_images, start=1):
        base64_img = encode_image(test_folder / img_name)
        if verbose:
            print(f"Encoding test image {i}: {img_name}")
        content.append({"type": "input_text", "text": f"### test{i}"})
        content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_img}", "detail": "high"})

    # Prompt part 3/3
    content.append({"type": "input_text",
                    "text": f"\n## Classify each of the test images whether it belongs to step {step_name}. Return a JSON object with indent matching the given schema.\n\
Be sure to check all the characteristics of each step of the assembly process again to make sure that your decision is correct before you respond."})

    messages.append({"role": "user", "content": content})

    if verbose:
        start_time = time.time()
        print("started parsing response")

    response = client.responses.parse(
        input=messages,
        model="gpt-5",
        reasoning={"effort": "minimal", "summary": "auto"},
        text_format=StepResponseList
    )

    if verbose:
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

    print(response.output_text)

    data = json.loads(response.output_text)
#    result = "yes" if any(r["classification"] == "yes" for r in data["responses"]) else "no"
#    result = "no" if any(r["classification"] == "no" for r in data["responses"]) else "yes"

    yes_votes = sum(1 for r in data["responses"]
                    if r["classification"] == "yes" and r["confidence"] >= 3)  # 4 or 5のみカウント
    result = "yes" if yes_votes >= 2 else "no"


    print("Vote result for step '{}': {}\n".format(step_name, result))
    return result

def main():
    parser = argparse.ArgumentParser(description="Run step classification for chair assembly images")
    parser.add_argument("--step_name", type=str, default="5_hexwrench_tight")
    parser.add_argument("--test_images", nargs='+',
                        default=["frame_0063_t63.0s.jpg", "frame_0064_t64.0s.jpg", "frame_0065_t65.0s.jpg"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    return run(args.step_name, args.test_images, args.verbose)

if __name__ == "__main__":
    main()
