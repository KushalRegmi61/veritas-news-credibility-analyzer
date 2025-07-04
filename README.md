<<<<<<< HEAD
<<<<<<< HEAD
# misinfo-detector
 Misinfo Detector — A Hybrid ML System for Detecting Fake News Articles using Text and Metadata
=======
# Misinformation Detector

A FastAPI-based service for detecting misinformation in text using a machine learning model.

## Structure
```
misinfo-detector/
├── .env                           # Environment variables
├── .gitignore                     # Git ignore file
├── app.py                         # Main Flask/FastAPI application
├── Dockerfile                     # Docker configuration
├── LICENSE                        # MIT License
├── output.png                     # Output visualization
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── api/                           # API-related files
│   ├── dependencies.py            # API dependencies
│   ├── main.py                    # API main entry point
│   ├── __pycache__/              # Python cache
│   └── schema/                    # API schema definitions
├── config/                        # Configuration files
│   └── feature_config.py          # Feature configuration
├── data/                          # Data directory
│   ├── processed/                 # Processed datasets
│   └── raw/                       # Raw datasets (True.csv, Fake.csv)
├── deployment/                    # Deployment configurations
│   ├── ec2_setup.md              # AWS EC2 setup guide
│   └── run_server.sh             # Server startup script
├── model/                         # Model artifacts
│   ├── __init__.py               # Package initialization
│   ├── best_params.pkl           # Best hyperparameters
│   ├── encoder.pkl               # Label encoder
│   ├── misinfo_detection_pipeline.pkl # Complete pipeline
│   ├── model_pipeline.pkl        # Model pipeline
│   ├── model.pkl                 # Trained model
│   ├── predict_output.py         # Prediction utilities
│   ├── text_preprocessor.pkl     # Text preprocessing pipeline
│   ├── vectorizer.pkl            # Text vectorizer
│   └── __pycache__/              # Python cache
├── notebooks/                     # Jupyter notebooks
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_modeling.ipynb         # Model training and evaluation
│   ├── eda.ipynb                 # Additional EDA
│   └── ...                       # Other notebooks
├── reports/                       # Analysis reports and figures
│   └── figures/                  # EDA visualizations
├── utils/                         # Utility functions
│   ├── model_utils.py            # Model handling utilities
│   └── preprocessing.py          # Text preprocessing utilities
```
## Quickstart
1. Install requirements: `pip install -r requirements.txt`
2. Run server: `uvicorn api.main:app --reload`
>>>>>>> 7dc618c (initial commit)
=======
>>>>>>> 6b6cb4c (update readme.md file)
