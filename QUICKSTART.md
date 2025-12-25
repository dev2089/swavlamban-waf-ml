# Quick Start Guide - WAF ML

Welcome to the WAF Machine Learning project! This guide will help you get up and running quickly.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setup Steps](#setup-steps)
3. [Command Reference](#command-reference)
4. [Directory Structure](#directory-structure)
5. [Feature Demonstrations](#feature-demonstrations)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Python**: 3.8 or higher
  - Verify: `python --version`
- **Git**: For version control
  - Verify: `git --version`
- **pip**: Python package manager (comes with Python)
  - Verify: `pip --version`
- **Virtual Environment**: Python's venv module (usually included)
  - Test: `python -m venv --help`
- **Required System Libraries** (Linux/Ubuntu):
  ```bash
  sudo apt-get install python3-dev build-essential
  ```
- **Disk Space**: At least 2GB free for dependencies and models
- **RAM**: Minimum 4GB recommended
- **Internet Connection**: For downloading packages and pre-trained models

---

## Setup Steps

### 1. Clone the Repository

```bash
git clone https://github.com/dev2089/swavlamban-waf-ml.git
cd swavlamban-waf-ml
```

### 2. Create a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Optional: Install development dependencies
pip install -r requirements-dev.txt
```

### 4. Verify Installation

```bash
python -c "import tensorflow; import sklearn; print('✓ Installation successful')"
```

### 5. Configure Environment Variables (Optional)

Create a `.env` file in the project root:

```bash
# .env file example
MODEL_PATH=./models
DATA_PATH=./data
LOG_LEVEL=INFO
```

Load environment variables:
```bash
source .env  # On macOS/Linux
# or
set -a; source .env; set +a  # Bash alternative
```

---

## Command Reference

### Training Commands

**Train a new model:**
```bash
python train.py --config configs/default.yaml --epochs 50 --batch-size 32
```

**Train with specific parameters:**
```bash
python train.py \
  --model-type transformer \
  --learning-rate 0.001 \
  --validation-split 0.2 \
  --save-checkpoint
```

**Resume training from checkpoint:**
```bash
python train.py --checkpoint models/latest_checkpoint.pth --continue-training
```

### Evaluation Commands

**Evaluate model performance:**
```bash
python evaluate.py --model models/trained_model.h5 --test-data data/test_set.csv
```

**Generate evaluation report:**
```bash
python evaluate.py --model models/trained_model.h5 --generate-report --output reports/
```

### Inference Commands

**Run inference on single sample:**
```bash
python predict.py --model models/trained_model.h5 --input sample.json
```

**Batch inference:**
```bash
python predict.py --model models/trained_model.h5 --batch-input data/samples.csv --output predictions.json
```

### Data Processing Commands

**Prepare dataset:**
```bash
python scripts/prepare_data.py --input raw_data/ --output processed_data/ --split 0.8
```

**Generate synthetic data:**
```bash
python scripts/generate_synthetic_data.py --samples 10000 --output synthetic_data/
```

### Utility Commands

**View model architecture:**
```bash
python -m utils.model_inspect --model models/trained_model.h5
```

**Check system compatibility:**
```bash
python scripts/check_env.py
```

**Run tests:**
```bash
pytest tests/ -v --cov=src/
```

---

## Directory Structure

```
swavlamban-waf-ml/
├── README.md                 # Project overview
├── QUICKSTART.md            # This file
├── requirements.txt         # Python dependencies
├── requirements-dev.txt     # Development dependencies
├── setup.py                 # Package setup configuration
├── .gitignore              # Git ignore rules
├── .env.example            # Environment variables template
│
├── src/                     # Source code
│   ├── __init__.py
│   ├── models/             # Model architectures
│   │   ├── __init__.py
│   │   ├── transformer.py
│   │   ├── lstm.py
│   │   └── hybrid.py
│   ├── data/               # Data processing modules
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   └── augmentation.py
│   ├── utils/              # Utility functions
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── visualization.py
│   │   └── helpers.py
│   └── config/             # Configuration handling
│       ├── __init__.py
│       └── settings.py
│
├── configs/                # Configuration files
│   ├── default.yaml
│   ├── production.yaml
│   └── experiments.yaml
│
├── data/                   # Data directory (git-ignored)
│   ├── raw/               # Raw data
│   ├── processed/         # Processed data
│   ├── train/             # Training data
│   ├── validation/        # Validation data
│   └── test/              # Test data
│
├── models/                # Trained models (git-ignored)
│   ├── checkpoints/       # Training checkpoints
│   └── final/             # Final trained models
│
├── notebooks/             # Jupyter notebooks
│   ├── exploration.ipynb
│   ├── training.ipynb
│   └── evaluation.ipynb
│
├── scripts/               # Standalone scripts
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── prepare_data.py
│   ├── generate_synthetic_data.py
│   └── check_env.py
│
├── tests/                 # Unit and integration tests
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_data.py
│   ├── test_utils.py
│   └── fixtures/
│
├── reports/              # Generated reports (git-ignored)
│   ├── metrics/
│   ├── visualizations/
│   └── logs/
│
└── docs/                 # Documentation
    ├── API.md
    ├── CONTRIBUTING.md
    └── ARCHITECTURE.md
```

---

## Feature Demonstrations

### 1. Quick Model Training

Get a model trained in minutes:

```bash
# Prepare sample data
python scripts/prepare_data.py --input data/raw --output data/processed --split 0.8

# Train a basic model
python train.py --epochs 10 --batch-size 32 --quick-mode

# Expected output: Model trained and saved to models/quick_model.h5
```

### 2. Running Inference

Use a trained model to make predictions:

```bash
# Single prediction
python predict.py --model models/quick_model.h5 --input '{"features": [1, 2, 3, 4, 5]}'

# Batch predictions
python predict.py --model models/quick_model.h5 --batch-input data/samples.csv --output results.json
```

### 3. Model Evaluation

Comprehensive model evaluation:

```bash
# Evaluate model
python evaluate.py --model models/quick_model.h5 --test-data data/test --generate-report

# Check outputs in reports/ directory
ls -la reports/
```

### 4. Jupyter Notebook Exploration

Interactive exploration and experimentation:

```bash
# Start Jupyter
jupyter notebook notebooks/

# Open exploration.ipynb to:
# - Load and visualize data
# - Experiment with models
# - Analyze results
```

### 5. Running Tests

Verify everything is working:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=src/ --cov-report=html
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Virtual Environment Not Activating

**Problem:** `source venv/bin/activate` command not found

**Solution:**
```bash
# Recreate virtual environment
rm -rf venv/
python -m venv venv
source venv/bin/activate
```

#### 2. Package Installation Fails

**Problem:** `pip install` command fails with permission errors

**Solution:**
```bash
# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Try installing with --user flag
pip install --user -r requirements.txt

# Or use cache-dir
pip install --no-cache-dir -r requirements.txt
```

#### 3. Out of Memory During Training

**Problem:** `MemoryError` or `CUDA out of memory`

**Solution:**
```bash
# Reduce batch size
python train.py --batch-size 8  # Instead of 32

# Enable gradient checkpointing
python train.py --gradient-checkpointing

# Use mixed precision training
python train.py --mixed-precision
```

#### 4. GPU Not Detected

**Problem:** TensorFlow/PyTorch not using GPU

**Solution:**
```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# For PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# Install GPU-specific packages
pip install tensorflow-gpu  # or torch with CUDA
```

#### 5. Data Loading Issues

**Problem:** `FileNotFoundError` when loading data

**Solution:**
```bash
# Verify data directory exists
ls -la data/raw/
ls -la data/processed/

# Check file permissions
chmod 644 data/raw/*

# Prepare data properly
python scripts/prepare_data.py --input data/raw --output data/processed
```

#### 6. Import Errors

**Problem:** `ModuleNotFoundError` for custom modules

**Solution:**
```bash
# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in development mode
pip install -e .
```

#### 7. Port Already in Use (Jupyter)

**Problem:** `Address already in use` when starting Jupyter

**Solution:**
```bash
# Specify different port
jupyter notebook --port 8889

# Or kill existing process
lsof -i :8888
kill -9 <PID>
```

### Diagnostic Commands

```bash
# Check Python version
python --version

# Verify all dependencies
pip list

# Check GPU/CUDA availability
nvidia-smi

# View system information
python scripts/check_env.py

# Run tests with verbose output
pytest tests/ -vv -s
```

### Getting Help

- **Documentation**: Check `docs/` directory
- **Issues**: Review GitHub issues for similar problems
- **Logs**: Check `reports/logs/` for error details
- **Community**: Open a new GitHub issue with:
  - Python version
  - OS and version
  - Full error message
  - Steps to reproduce

---

## Next Steps

1. ✅ Complete the setup steps above
2. 📚 Read the [README.md](README.md) for project overview
3. 🚀 Try the [Feature Demonstrations](#feature-demonstrations)
4. 📖 Explore [API Documentation](docs/API.md)
5. 🧪 Run tests with `pytest tests/`
6. 📓 Check out example [Jupyter notebooks](notebooks/)
7. 🔧 Customize configurations in [configs/](configs/)

---

## Quick Troubleshooting Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `python scripts/check_env.py` runs successfully
- [ ] Sample training completes without errors
- [ ] Tests pass with `pytest tests/`

---

## Version Information

- **Project Version**: 1.0.0
- **Last Updated**: 2025-12-25
- **Maintained By**: dev2089

For the latest updates, visit the [GitHub repository](https://github.com/dev2089/swavlamban-waf-ml).

---

**Happy Machine Learning! 🚀**
