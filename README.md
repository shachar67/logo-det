# Logo Detection Project

This project implements a logo detection system using YOLOv8, capable of detecting various logos in images.

For the full project report, see the [PDF report](results/LogoDetection_Report.pdf).

## Project Structure

```
logo-det/
├── main.py                   # Unified CLI entry point
├── data/                     # Raw dataset directory
│   ├── logos-dataset-section-1/
│   └── logos-dataset-section-2/
├── src/                      # Source code
│   ├── dataset/             # Dataset preprocessing utilities
│   │   ├── preprocess.py    # Main preprocessing script
│   │   ├── class_scanner.py
│   │   ├── label_converter.py
│   │   ├── dataset_validator.py
│   │   └── analyze.py       # Dataset analysis & visualization
│   └── scripts/             # Training/evaluation/prediction
│       ├── base.py          # Base class for scripts
│       ├── trainer.py       # Training logic
│       ├── evaluator.py     # Evaluation logic
│       └── predictor.py     # Prediction logic
├── weights/                  # Model weights (downloaded/saved here)
├── configs/                  # Configuration files
│   └── data.yaml            # Auto-generated dataset config
├── results/                  # Model outputs and evaluations
├── requirements.txt
└── README.md
```

## Setup

1. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Data Setup:
   - Download the dataset from the provided source.
   - Unzip the contents into the `data/` directory.
   - The final structure should look like this:
     ```
     data/
     ├── logos-dataset-section-1/
     └── logos-dataset-section-2/
     ```

## Usage

### 1. Data Preparation (Required - Run Once)

This script will:
- Verify dataset structure
- Automatically detect all logo classes
- Create data.yaml configuration
- Convert label files from class names to numeric indices

```bash
python src/dataset/preprocess.py
```

### 2. Training

Train the model:
```bash
python main.py train --epochs 50 --batch-size 16
```

### 3. Evaluation

Evaluate trained model:
```bash
python main.py evaluate --weights logo_detection/train/weights/best.pt
```

### 4. Prediction

Run predictions:
```bash
python main.py predict --weights best.pt --source image.jpg --save
```

### 5. Zero-Shot Prediction (Any Logo) - Section 2

Run zero-shot prediction on any image using a powerful generative model. This command can identify logos that the model was not explicitly trained on.
> **Note:**  
> Zero-shot logo prediction uses a generative AI API. For larger images or complex scenes, the process may take a few seconds to several minutes, depending on image size, content, and your network speed. Please be patient while waiting for results.

```bash
python main.py any_logo --source path/to/your/image.jpg
```

You can also provide a ground truth label file for an immediate evaluation of the prediction:

```bash
python main.py any_logo --source path/to/your/image.jpg --labels path/to/your/labels.txt
```

### CLI Commands

The unified CLI supports four main commands:

**Train:**
```bash
python main.py train [OPTIONS]
  --data          Path to data.yaml (default: configs/data.yaml)
  --epochs        Number of epochs (default: 50)
  --batch-size    Batch size (default: 8)
  --img-size      Image size (default: 640)
  --weights       Initial weights (default: weights/yolov8n.pt)
  --lr            Learning rate (default: 0.001)
  --patience      Early stopping patience (default: 10)
  --device        Device (e.g., 0 or cpu)
```

**Evaluate:**
```bash
python main.py evaluate --weights WEIGHTS [OPTIONS]
  --data          Path to data.yaml (default: configs/data.yaml)
  --iou-thres     IOU threshold (default: 0.5)
  --conf-thres    Confidence threshold (default: 0.25)
  --save-json     Save results to JSON
```

**Predict:**
```bash
python main.py predict --weights WEIGHTS --source SOURCE [OPTIONS]
  --conf-thres    Confidence threshold (default: 0.25)
  --iou-thres     NMS IOU threshold (default: 0.45)
  --save          Save images with predictions
  --show          Display results
  --save-txt      Save results to txt files
```

**Any Logo (Zero-Shot):**
```bash
python main.py any_logo --source SOURCE [OPTIONS]
  --labels        Optional path to ground truth labels for evaluation
```

Get help for any command:
```bash
python main.py --help
python main.py train --help
python main.py evaluate --help
python main.py predict --help
python main.py any_logo --help
```

## Project Components

1. Baseline Model
   - YOLOv8 implementation for logo detection
   - Training and evaluation pipeline

2. Improvements
   - High resolution model

3. New Logo Detection
   - Solution for detecting new logos from section 2

## Results

Detailed results and analysis can be found in the project report.
