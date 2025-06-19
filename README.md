Image Sharpness Classifier
A Python tool for classifying images as sharp or blurry using traditional computer vision algorithms. No machine learning training required.

🚀 Quick Start
Installation
bash
pip install -r requirements.txt
Basic Usage
bash
# Simple classification
python main.py --source /path/to/images --output /path/to/results

# Using configuration file
python main.py --config config.json

# Interactive mode
python main.py
📁 Input Structure
Your source folder should be organized with time-format directories containing image-JSON pairs:

source_folder/
├── 2024-01-01/
│   ├── img001.jpg
│   ├── img001.json
│   ├── img002.png
│   └── img002.json
├── 2024-01-02/
│   ├── photo001.jpg
│   ├── photo001.json
│   └── ...
└── ...
Supported time formats:

2024-01-01, 2024_01_01, 20240101
2024-01-01_12-30-45, 20240101123045
📤 Output Structure
output_folder/
├── 2024-01-01/
│   ├── Sharp/          # Sharp images + JSON files
│   └── Blurry/         # Blurry images + JSON files
├── 2024-01-02/
│   ├── Sharp/
│   └── Blurry/
├── processing_report.json
├── processing_results.csv
└── quality_distribution.png
⚙️ Configuration & Hyperparameters
Create config.json to customize settings:

json
{
  "source_root": "./input_images",
  "output_root": "./classified_images",
  "classifier_params": {
    "laplacian_threshold": 100.0,
    "sobel_threshold": 50.0,
    "brenner_threshold": 1000.0,
    "tenengrad_threshold": 500.0
  },
  "processing": {
    "max_workers": 4,
    "supported_formats": [".jpg", ".jpeg", ".png", ".bmp"]
  },
  "reports": {
    "generate_visual": true,
    "export_csv": true
  }
}
🎛️ Tuning Thresholds
The tool uses 4 algorithms that vote on image sharpness. Adjust thresholds based on your image type:

For natural photos (default):

json
"classifier_params": {
  "laplacian_threshold": 100.0,
  "sobel_threshold": 50.0,
  "brenner_threshold": 1000.0,
  "tenengrad_threshold": 500.0
}
For portraits (more sensitive):

json
"classifier_params": {
  "laplacian_threshold": 80.0,
  "sobel_threshold": 40.0,
  "brenner_threshold": 800.0,
  "tenengrad_threshold": 400.0
}
For documents/text (stricter):

json
"classifier_params": {
  "laplacian_threshold": 150.0,
  "sobel_threshold": 70.0,
  "brenner_threshold": 1500.0,
  "tenengrad_threshold": 700.0
}
📊 How It Works
Four algorithms analyze each image: Laplacian, Sobel, Brenner, Tenengrad
Each algorithm votes "sharp" or "blurry" based on its threshold
Majority vote determines final classification
Higher thresholds = stricter classification (fewer images classified as sharp)
🛠️ Command Line Options
bash
python main.py \
  --source ./images \
  --output ./results \
  --max-workers 8 \          # Parallel processing threads
  --preview \                # Preview mode (no processing)
  --max-folders 3 \          # Limit folders in preview
  --no-visual \              # Skip visualization
  --no-csv                   # Skip CSV export
📋 Requirements
Python 3.7+
OpenCV
NumPy
SciPy
tqdm
matplotlib (optional, for reports)
pandas (optional, for CSV export)

