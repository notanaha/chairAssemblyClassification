#!/usr/bin/env python3
# Identical to **stepDetection.py** 
# 
"""Run chair assembly step detection across all (or selected) steps in parallel.

Examples (PowerShell):
  python 0_stepDetection.py --test-images frame_0053_t53.0s.jpg frame_0054_t54.0s.jpg frame_0055_t55.0s.jpg
  python 0_stepDetection.py --steps 4_hexwrench_light 5_hexwrench_tight --test-images frame_0063_t63.0s.jpg frame_0064_t64.0s.jpg --max-concurrency 2

Notes:
- Concurrency uses asyncio + threads because chairAssemblyStep.run performs blocking network I/O.
- Adjust --max-concurrency to stay within Azure OpenAI rate limits.
"""
from __future__ import annotations
import argparse
import asyncio
import os
import time
from typing import List, Sequence, Tuple
from dotenv import load_dotenv

# Local import
from utils import chairAssemblyStep

ALL_STEPS = [
    "1_pickupbolt",
    "2_placeparts",
    "3_placebolts",
    "4_hexwrench_light",
    "5_hexwrench_tight",
    "6_attachcylinder",
    "7_moveseat",
    "8_attachleg",
    "9_finishedchair",
]

async def _run_one(step: str, images: Sequence[str], verbose: bool, sem: asyncio.Semaphore) -> Tuple[str, str, float]:
    start = time.time()
    async with sem:
        # Execute blocking function in a thread to achieve parallelism
        result = await asyncio.to_thread(chairAssemblyStep.run, step, list(images), verbose)
    elapsed = time.time() - start
    return step, result, elapsed

async def run_parallel(steps: Sequence[str], images: Sequence[str], verbose: bool, max_concurrency: int) -> list[Tuple[str, str, float]]:
    sem = asyncio.Semaphore(max_concurrency)
    tasks = [asyncio.create_task(_run_one(s, images, verbose, sem)) for s in steps]
    results: List[Tuple[str, str, float]] = []
    for coro in asyncio.as_completed(tasks):
        try:
            results.append(await coro)
        except Exception as e:  # noqa: BLE001
            step_name = "<unknown>"
            print(f"[ERROR] {step_name}: {e}")
    return results

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel step detection for chair assembly images")
    parser.add_argument("--steps", nargs="*", default=ALL_STEPS, help="Subset of step names to run (default: all)")
    parser.add_argument("--test-images", nargs="+", required=True, help="Test image filenames located under output/ directory")
    parser.add_argument("--max-concurrency", type=int, default=4, help="Maximum concurrent step classifications (default: 4)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging passed to underlying classifier")
    return parser.parse_args()

async def async_main():
    args = parse_args()

    load_dotenv(override=True)

    steps = args.steps
    images = args.test_images
    unknown = [s for s in steps if s not in ALL_STEPS]
    if unknown:
        raise SystemExit(f"Unknown step names: {', '.join(unknown)}")

    print(f"Running {len(steps)} step classifications with max_concurrency={args.max_concurrency}")
    print("Test images:", ", ".join(images))
    overall_start = time.time()

    results = await run_parallel(steps, images, args.verbose, args.max_concurrency)
    overall_elapsed = time.time() - overall_start

    # Sort results by step order (numeric prefix)
    ordering = {name: i for i, name in enumerate(ALL_STEPS)}
    results.sort(key=lambda x: ordering.get(x[0], 999))

    print("\n=== Summary ===")
    for step, result, elapsed in results:
        print(f"{step:20s} -> {result}  ({elapsed:.1f}s)")

    yes_steps = [s for s, r, _ in results if r == "yes"]
    print("\nSteps classified as YES:")
    if yes_steps:
        for s in yes_steps:
            print(f"  - {s}")
    else:
        print("  (none)")

    print(f"\nTotal elapsed: {overall_elapsed:.1f}s for {len(results)} steps")

def main():  # synchronous entrypoint
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("Interrupted.")

if __name__ == "__main__":
    main()
