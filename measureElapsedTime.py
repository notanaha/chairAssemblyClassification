#!/usr/bin/env python3
"""
Measure elapsed time and classify assembly steps for chair assembly process.

This script processes frames in non-overlapping triplets and classifies assembly steps
using the stepDetection.py module.
"""

import os
import re
import asyncio
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import time
from collections import defaultdict

# Import the existing step detection functionality
from stepDetection import run_parallel, ALL_STEPS

# Define paths
BASE_DIR = Path(__file__).resolve().parent
UTILS_DIR = BASE_DIR / "utils"
FRAMES_LIST_PATH = UTILS_DIR / "0_listofFrames.txt"
OUTPUT_DIR = BASE_DIR / "output"

def load_frame_list() -> List[str]:
    """Load the list of frames from the text file."""
    try:
        with open(FRAMES_LIST_PATH, 'r', encoding='utf-8') as f:
            frames = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(frames)} frames from {FRAMES_LIST_PATH}")
        return frames
    except FileNotFoundError:
        raise FileNotFoundError(f"Frame list file not found: {FRAMES_LIST_PATH}")

def extract_timestamp(frame_name: str) -> float:
    """Extract timestamp from frame name.
    
    Example: frame_0001_t1.0s.jpg -> 1.0
    """
    match = re.search(r't(\d+\.?\d*)s', frame_name)
    if match:
        return float(match.group(1))
    else:
        # Fallback: try to extract from frame number
        match = re.search(r'frame_(\d+)_', frame_name)
        if match:
            return float(match.group(1))
        return 0.0

def extract_timestamp_suffix(frame_name: str) -> str:
    """Extract timestamp suffix from frame name.
    
    Example: frame_0001_t1.0s.jpg -> t1.0s
    """
    match = re.search(r't(\d+\.?\d*s)', frame_name)
    if match:
        return match.group(1)
    else:
        # Fallback
        timestamp = extract_timestamp(frame_name)
        return f't{timestamp}s'

def create_triplets(frames: List[str]) -> List[List[str]]:
    """Create non-overlapping triplets from the frame list."""
    triplets = []
    for i in range(0, len(frames), 3):
        triplet = frames[i:i+3]
        if len(triplet) == 3:  # Only process complete triplets
            triplets.append(triplet)
        elif len(triplet) > 0:  # Handle remaining frames
            # Pad with the last frame if needed, or handle as incomplete
            print(f"Warning: Incomplete triplet at end with {len(triplet)} frames: {triplet}")
            triplets.append(triplet)
    return triplets

async def classify_triplet(triplet: List[str], max_concurrency: int = 9, verbose: bool = False) -> Dict[str, str]:
    """Classify a triplet of frames for all assembly steps.
    
    Returns a dictionary mapping step names to 'yes'/'no' classifications.
    """
    try:
        # Check if all frames exist
        for frame in triplet:
            frame_path = OUTPUT_DIR / frame
            if not frame_path.exists():
                print(f"Warning: Frame not found: {frame_path}")
                return {step: "unknown" for step in ALL_STEPS}
        
        # Run step detection for all steps
        results = await run_parallel(ALL_STEPS, triplet, verbose, max_concurrency)
        
        # Convert results to dictionary
        step_results = {step: result for step, result, _ in results}
        return step_results
        
    except Exception as e:
        print(f"Error classifying triplet {triplet}: {e}")
        return {step: "unknown" for step in ALL_STEPS}

def calculate_triplet_timestamp(triplet: List[str]) -> float:
    """Calculate timestamp for a triplet using the middle frame."""
    if len(triplet) >= 2:
        # Use middle frame (index 1 for triplet of 3)
        middle_idx = len(triplet) // 2
        return extract_timestamp(triplet[middle_idx])
    elif len(triplet) == 1:
        return extract_timestamp(triplet[0])
    else:
        return 0.0

async def process_all_triplets(frames: List[str], max_concurrency: int = 9, verbose: bool = False) -> pd.DataFrame:
    """Process all triplets and return a DataFrame with results."""
    
    triplets = create_triplets(frames)
    print(f"Processing {len(triplets)} triplets...")
    
    results = []
    
    for i, triplet in enumerate(triplets):
        print(f"Processing triplet {i+1}/{len(triplets)}: {triplet}")
        
        # Classify the triplet
        step_classifications = await classify_triplet(triplet, max_concurrency, verbose)
        
        # Extract positive classifications
        positive_steps = [step for step, classification in step_classifications.items() 
                         if classification == "yes"]
        
        # If no positive classifications, mark as unknown
        if not positive_steps:
            positive_steps = ["unknown"]
        
        # Calculate timestamp
        timestamp = calculate_triplet_timestamp(triplet)
        
        # Extract frame timestamp suffixes
        frame_names = [extract_timestamp_suffix(frame) for frame in triplet]
        
        # Add to results
        results.append({
            'timestamp': timestamp,
            'step_names': positive_steps,
            'frame_names': frame_names,
            'triplet_frames': triplet  # Keep original frame names for reference
        })
        
        # Add a small delay to avoid overwhelming the API
        if i < len(triplets) - 1:  # Don't delay after the last triplet
            await asyncio.sleep(1)
    
    return pd.DataFrame(results)

def generate_summary(df: pd.DataFrame) -> Dict[str, int]:
    """Generate a summary of step counts."""
    step_counts = defaultdict(int)
    
    for step_list in df['step_names']:
        for step in step_list:
            step_counts[step] += 1
    
    return dict(step_counts)

async def main():
    """Main function to run the elapsed time measurement."""
    print("=== Chair Assembly Step Detection and Timing Analysis ===")
    
    # Record script start time
    script_start_time = time.time()
    
    # Load frame list
    frames = load_frame_list()
    
    if not frames:
        print("No frames found in the list.")
        return
    
    print(f"Total frames to process: {len(frames)}")
    
    # Process all triplets
    start_time = time.time()
    df = await process_all_triplets(frames, max_concurrency=9, verbose=False)
    end_time = time.time()
    
    print(f"\n=== Processing Complete ===")
    print(f"Total processing time: {end_time - start_time:.1f} seconds")
    
    # Display DataFrame
    print(f"\n=== Results DataFrame ===")
    print(df.to_string(index=False))
    
    # Generate summary
    summary = generate_summary(df)
    print(f"\n=== Summary of Step Classifications ===")
    for step, count in sorted(summary.items()):
        print(f"{step}: {count}")
    
    # Save results
    output_path = BASE_DIR / "step_analysis_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Save summary
    summary_path = BASE_DIR / "step_analysis_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Step Classification Summary\n")
        f.write("=" * 30 + "\n")
        for step, count in sorted(summary.items()):
            f.write(f"{step}: {count}\n")
        f.write(f"\nTotal processing time: {end_time - start_time:.1f} seconds\n")
        f.write(f"Total triplets processed: {len(df)}\n")
        f.write(f"Total frames processed: {len(frames)}\n")
    print(f"Summary saved to: {summary_path}")
    
    # Calculate and display total script execution time
    script_end_time = time.time()
    total_script_duration = script_end_time - script_start_time
    print(f"\n=== Total Script Execution Time ===")
    print(f"Total execution duration: {total_script_duration:.1f} seconds")
    
    return df, summary

if __name__ == "__main__":
    # Run the async main function
    try:
        import sys
        if sys.platform == 'win32':
            # On Windows, use ProactorEventLoop for better compatibility
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        df, summary = asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
