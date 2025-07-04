**Create virtual environment (choose one method)**
Method 1: Using venv
```bash
python -m venv venv

source venv/bin/activate
```

**Method 2: Using conda(recommend)**
```bash
conda create -n image-quality python=3.8
conda activate image-quality
```

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Quick start**
```bash
python main.py --config enhanced_config.json
```