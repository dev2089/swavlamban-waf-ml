# Swavlamban WAF ML

A machine learning solution for Web Application Firewall (WAF) detection and analysis using Swavlamban framework.

## Overview

This project implements machine learning models for Web Application Firewall (WAF) detection, utilizing the Swavlamban framework to identify and classify security threats in web traffic.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or higher
- pip (Python package manager)
- Git
- Virtual environment tool (venv or conda)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/dev2089/swavlamban-waf-ml.git
cd swavlamban-waf-ml
```

### 2. Create a Virtual Environment

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n swavlamban-waf-ml python=3.8
conda activate swavlamban-waf-ml
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file in the project root (if needed):
```bash
cp .env.example .env
```

2. Update the `.env` file with your configuration parameters:
```
LOG_LEVEL=INFO
DATA_PATH=./data
MODEL_PATH=./models
```

## Usage

### Running the Application

```bash
python main.py
```

### Running with Configuration

```bash
python main.py --config config.yaml --log-level DEBUG
```

### Running Tests

```bash
python -m pytest tests/
```

### Running with Coverage

```bash
python -m pytest tests/ --cov=src --cov-report=html
```

## Project Structure

```
swavlamban-waf-ml/
├── README.md
├── requirements.txt
├── setup.py
├── .env.example
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── waf_detector.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   └── test_data.py
├── data/
│   ├── raw/
│   └── processed/
└── models/
```

## Training Models

To train the WAF ML models:

```bash
python src/train.py --data data/processed --output models/
```

### Training Options

- `--data`: Path to processed training data
- `--output`: Output directory for trained models
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size for training (default: 32)
- `--validation-split`: Validation data split ratio (default: 0.2)

## Making Predictions

To use trained models for predictions:

```bash
python src/predict.py --model models/waf_model.pkl --input data/test_data.csv
```

## Data Format

Input data should be in CSV format with the following structure:

```
feature_1,feature_2,feature_3,...,feature_n,label
value1,value2,value3,...,valuen,attack_type
```

Supported attack types:
- SQL_INJECTION
- XSS
- DDOS
- NORMAL
- BRUTE_FORCE

## Troubleshooting

### Common Issues

**Issue: Module not found error**
```bash
# Solution: Ensure virtual environment is activated and dependencies installed
pip install -r requirements.txt
```

**Issue: Port already in use**
```bash
# Solution: Change the port in configuration or kill the process using the port
```

**Issue: Data not found**
```bash
# Solution: Ensure data files are in the correct directory (./data/)
```

## Performance Metrics

The models are evaluated using:

- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or suggestions, please:

- Open an issue on [GitHub Issues](https://github.com/dev2089/swavlamban-waf-ml/issues)
- Contact the maintainers

## Authors

- dev2089

## Changelog

### [1.0.0] - 2025-12-25
- Initial release
- Basic WAF ML detection model
- Training and prediction modules
- Comprehensive documentation

---

**Last Updated**: 2025-12-25
