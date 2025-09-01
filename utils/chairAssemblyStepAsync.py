# chairAssemblyStep01.py (async-enabled variant)
#!/usr/bin/env python3
"""Async variant of chairAssemblyStep with reusable clients and run_async()."""
import os, json, time, base64, glob, argparse
from typing import Literal, List
from pathlib import Path

import openai
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv(override=True)

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
PROMPT_DIR = BASE_DIR
OUTPUT_DIR = ROOT_DIR / "output"

_sync_client: openai.AzureOpenAI | None = None
_async_client: openai.AsyncAzureOpenAI | None = None

def _get_sync_client():
    global _sync_client
    if _sync_client is None:
        _sync_client = openai.AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )
    return _sync_client

def _get_async_client():
    global _async_client
    if _async_client is None:
        _async_client = openai.AsyncAzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )
    return _async_client

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

class StepResponse(BaseModel):
    classification: Literal["yes", "no"]
    confidence: int = Field(..., ge=1, le=5)
    reasoning: str

class StepResponseList(BaseModel):
    responses: List[StepResponse]
    overall_result: Literal["yes", "no"]
    reasoning: str

def _build_messages(step_name: str, test_images: List[str], verbose: bool):
    with open(PROMPT_DIR / f"{step_name}.txt", "r", encoding="utf-8") as f:
        user_prompt_template = f.read()
    if verbose:
        print(f"Using system message: {step_name}.txt")

    messages = []
    content = []
    content.append({"type": "input_text", "text": "You are analyzing the productivity of a chair assembly process. The following images are examples of the chair assembly process."})

    sample_folder = OUTPUT_DIR / step_name
    image_files = sorted(glob.glob(str(sample_folder / "*.jpg")))
    for idx, image_path in enumerate(image_files, start=1):
        image_name = os.path.basename(image_path)
        b64 = encode_image(image_path)
        if verbose:
            print(f"Encoding sample image {idx}: {image_name}")
        content.append({"type": "input_text", "text": f"### sample{idx}"})
        content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}", "detail": "high"})

    content.append({"type": "input_text", "text": user_prompt_template})

    test_folder = OUTPUT_DIR
    for i, img_name in enumerate(test_images, start=1):
        b64 = encode_image(test_folder / img_name)
        if verbose:
            print(f"Encoding test image {i}: {img_name}")
        content.append({"type": "input_text", "text": f"### test{i}"})
        content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}", "detail": "high"})

    content.append({"type": "input_text", "text": f"Classify each of the images whether it belongs to step {step_name}. Return a JSON object with indent matching the given schema."})
    messages.append({"role": "user", "content": content})
    return messages

def run(step_name: str, test_images: List[str], verbose: bool = True) -> str:
    client = _get_sync_client()
    messages = _build_messages(step_name, test_images, verbose)
    if verbose:
        start_time = time.time(); print("started parsing response")
    response = client.responses.parse(
        input=messages,
        model="gpt-5-mini",
        reasoning={"effort": "high", "summary": "auto"},
        text_format=StepResponseList,
    )
    if verbose:
        print(f"Elapsed time: {time.time()-start_time:.2f} s")
    print(response.output_text)
    data = json.loads(response.output_text)
    result = "yes" if any(r["classification"] == "yes" for r in data["responses"]) else "no"
    print("Classification result:", result, "\n")
    return result

async def run_async(step_name: str, test_images: List[str], verbose: bool = True) -> str:
    client = _get_async_client()
    messages = _build_messages(step_name, test_images, verbose)
    if verbose:
        start_time = time.time(); print("started parsing response (async)")
    response = await client.responses.parse(
        input=messages,
        model="gpt-5-mini",
        reasoning={"effort": "high", "summary": "auto"},
        text_format=StepResponseList,
    )
    if verbose:
        print(f"Elapsed time (async): {time.time()-start_time:.2f} s")
    print(response.output_text)
    data = json.loads(response.output_text)
    result = "yes" if any(r["classification"] == "yes" for r in data["responses"]) else "no"
    print("Classification result:", result, "\n")
    return result

def main():
    parser = argparse.ArgumentParser(description="Run step classification (async-capable variant)")
    parser.add_argument("--step_name", type=str, default="5_hexwrench_tight")
    parser.add_argument("--test_images", nargs='+', default=["frame_0063_t63.0s.jpg", "frame_0064_t64.0s.jpg", "frame_0065_t65.0s.jpg"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    return run(args.step_name, args.test_images, args.verbose)

if __name__ == "__main__":
    main()
