# Chair Assembly Step Detection System

This repository contains the scripts that processes video frames to identify and classify different assembly steps.

## Folder Structure

```
scripts/
├── Core Scripts
│   ├── stepDetection.py             # Parallel step classifier
│   ├── 0_chairAssembly_01respApi.py # Single-step classifier
│   └── measureElapsedTime.py        # End-to-end analysis pipeline
│   
├── Utils & Configuration
│    └── utils/
│        ├── chairAssemblyStep.py    # Core classification engine
│        ├── {step_name}.txt         # Step-specific prompts (9 files)
│        └── 0_listofFrames.txt      # Frame sequence list
│
└── output
     ├── {step_name}/                # Sample frames presented to the classifier
     └── **.jpg                      # Test frames presented to the classifier

```


## Overview

The chair assembly process consists of 9 distinct steps:

1. **1_pickupbolt**: Pick up 4 bolts from a box
2. **2_placeparts**: Place the base plate onto the cushion  
3. **3_placebolts**: Place 4 bolts onto the base plate
4. **4_hexwrench_light**: Lightly screw the bolts using a hex wrench
5. **5_hexwrench_tight**: Firmly tighten the bolts using a hex wrench
6. **6_attachcylinder**: Attach a cylinder to the base plate
7. **7_moveseat**: Move the cushion with the base plate into position for leg attachment
8. **8_attachleg**: Attach legs to the assembly
9. **9_finishedchair**: Complete chair assembly

## Usage

### `stepDetection.py`
Input **--test-images** takes a triplet of consecutive frames from a video.

```bash
python stepDetection.py --test-images frame_0003_t3.0s.jpg frame_0004_t4.0s.jpg frame_0005_t5.0s.jpg
```

**Purpose**: Parallel step detection engine for chair assembly images

**Key Features**:
- **Asynchronous Processing**: Uses asyncio + threading for concurrent step classification
- **Rate Limiting**: Configurable concurrency control to respect Azure OpenAI API limits
- **Comprehensive Coverage**: Tests all 9 assembly steps simultaneously
- **Robust Error Handling**: Graceful handling of API failures and timeouts

**Command Line Arguments**:
- `--steps`: Subset of step names to run (default: all 9 steps)
- `--test-images`: Test image filenames (required, located in `output/` directory)
- `--max-concurrency`: Maximum concurrent classifications (default: 4)
- `--verbose`: Enable detailed logging

**Output**:
- Individual step classifications ("yes"/"no")
- Processing time for each step
- Summary of positive classifications
- Total elapsed time

### `measureElapsedTime.py`

```bash
python measureElapsedTime.py
```

**Purpose**: End-to-end analysis pipeline for processing video frame sequences

**Key Features**:
- **Triplet Processing**: Processes frames in non-overlapping groups of 3 for robust classification
- **Temporal Analysis**: Extracts timestamps and maintains temporal sequence
- **Batch Processing**: Handles large frame sequences efficiently
- **Data Export**: Saves results in CSV format for further analysis
- **Comprehensive Reporting**: Generates detailed summaries and statistics

**Workflow**:
1. **Frame Loading**: Reads frame list from `utils/0_listofFrames.txt`
2. **Triplet Creation**: Groups frames into non-overlapping triplets
3. **Step Classification**: Uses `stepDetection.py` to classify each triplet
4. **Timestamp Extraction**: Extracts timing information from frame names
5. **Result Aggregation**: Compiles results into structured format
6. **Export**: Saves to CSV and generates summary reports

**Key Functions**:
- `load_frame_list()`: Loads frame sequence from configuration file
- `extract_timestamp()`: Parses timing from frame filenames (e.g., "frame_0001_t1.0s.jpg" → 1.0)
- `create_triplets()`: Groups frames into processing batches
- `classify_triplet()`: Runs step detection on frame groups
- `process_all_triplets()`: Main processing pipeline

**Output Files**:
- `step_analysis_results.csv`: Detailed results with timestamps, classifications, and frame references
- `step_analysis_summary.txt`: Aggregated statistics and processing metrics

**Data Structure**:
```
timestamp | step_names | frame_names | triplet_frames
1.0       | ['1_pickupbolt'] | ['t0.0s', 't1.0s', 't2.0s'] | ['frame_0000_t0.0s.jpg', ...]
4.0       | ['2_placeparts', '4_hexwrench_light'] | [...] | [...]
```

### `utils/chairAssemblyStep.py`

**Purpose**: Core AI-powered step classification engine

**Key Features**:
- **Azure OpenAI Integration**: Uses GPT-5 with structured output parsing
- **Multi-modal Analysis**: Processes both sample images and test images
- **Confidence Scoring**: 1-5 scale confidence ratings for each classification
- **Voting System**: Requires ≥2 confident "yes" votes for positive classification
- **Base64 Encoding**: Handles image encoding for API transmission

**Classification Process**:
1. **Sample Loading**: Loads reference images for each step from `output/{step_name}/` directories
2. **Prompt Loading**: Reads step-specific prompts from `utils/{step_name}.txt`
3. **Image Encoding**: Converts images to base64 for API transmission
4. **AI Analysis**: Sends multi-modal prompt to Azure OpenAI GPT-5
5. **Structured Response**: Parses JSON response with classification and confidence
6. **Voting Logic**: Applies confidence thresholds for final decision

**Response Format**:
```json
{
  "responses": [
    {
      "classification": "yes",
      "confidence": 4,
      "reasoning": "Clear hex wrench tightening motion visible"
    }
  ],
  "overall_result": "yes",
  "reasoning": "Multiple confident positive classifications"
}
```

**Environment Requirements**:
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI service endpoint
- `AZURE_OPENAI_API_KEY`: Authentication key
- `AZURE_OPENAI_API_VERSION`: API version specification

**Decision Logic**:
- Counts "yes" votes with confidence ≥ 3
- Requires ≥ 2 confident votes for positive classification
- Returns "yes" or "no" based on voting threshold

## System Architecture

```
Video Frames → measureElapsedTime.py → stepDetection.py → chairAssemblyStep.py → Azure OpenAI
                      ↓                        ↓                    ↓
              Triplet Processing      Parallel Execution    AI Classification
                      ↓                        ↓                    ↓
              step_analysis_results.csv ← Result Aggregation ← Confidence Voting
```

## Dependencies

- **Python 3.8+**
- **asyncio**: Asynchronous processing
- **pandas**: Data manipulation and CSV export
- **openai**: Azure OpenAI client
- **pydantic**: Structured data validation
- **python-dotenv**: Environment variable management
- **pathlib**: Cross-platform path handling

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   Create `.env` file with:
   ```
   AZURE_OPENAI_ENDPOINT=your_endpoint
   AZURE_OPENAI_API_KEY=your_key
   AZURE_OPENAI_API_VERSION=your_version
   ```

3. **Prepare Data**:
   - Place frame images in `output/` directory
   - Create frame list in `utils/0_listofFrames.txt`
   - Ensure sample images exist in `output/{step_name}/` directories

## Usage Workflow

1. **Extract frames** from assembly video
2. **Run full analysis**:
   ```bash
   python measureElapsedTime.py
   ```
3. **Or test specific frames**:
   <br>--test-images must be a triplet of frames
   ```bash
   python stepDetection.py --test-images frame_0003_t3.0s.jpg frame_0004_t4.0s.jpg frame_0005_t5.0s.jpg
   ```
4. **Analyze results** using generated CSV files

## Performance Considerations

- **Concurrency**: Adjust `--max-concurrency` based on API rate limits
- **Batch Size**: Triplet processing balances accuracy and efficiency  
- **API Costs**: Monitor Azure OpenAI usage for large datasets
- **Processing Time**: Expect ~2-5 seconds per step classification

## Troubleshooting

- **API Rate Limits**: Reduce `max_concurrency` parameter
- **Missing Images**: Verify file paths and directory structure
- **Environment Issues**: Check `.env` file configuration
- **Memory Usage**: Process large datasets in smaller batches

## Output Analysis

The system generates detailed analytics including:
- Temporal progression of assembly steps
- Step detection confidence scores
- Processing performance metrics
- Visual scatter plots (via `step_analysi_results.ipynb`)
