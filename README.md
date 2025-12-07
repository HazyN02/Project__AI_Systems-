# FraudGen: Synthetic Fraud Data Generation & Detection System

FraudGen is an end-to-end system for generating synthetic fraud data,
engineering features, training fraud detection models, and evaluating
their performance. The project includes CTGAN-based synthetic data
generation, traditional machine learning models, evaluation reports, and
an application interface for inference.

## Project Structure

    Final FraudGen.ipynb
    Fraudgen demo 1.mp4
    Plots.zip
    Reports.zip
    app.py
    features.json
    final_rf.pkl
    fraudgen_qt.pkl
    iforest.pkl
    label_encoders.pkl
    model_features.pkl
    quantile_transformer.pkl
    top_features.json
    top_features_ctgan.pkl
    tstr_metrics.csv
    validation_summary.csv

## Overview

Fraud detection is often limited by the scarcity and imbalance of
real-world datasets. FraudGen addresses this by:

-   Generating realistic synthetic fraud data using CTGAN
-   Training fraud detection models (Random Forest, Isolation Forest,
    Quantile-based models)
-   Evaluating with TSTR (Train on Synthetic, Test on Real) metrics
-   Providing validation summaries, plots, and reports
-   Offering an application interface (app.py) for prediction

## Features

-   Synthetic Data Generation: CTGAN-based fraud dataset generation
-   Fraud Detection Models: Random Forest, Isolation Forest, Quantile
    transformation workflows
-   Evaluation: TSTR scores, validation metrics, summary reports
-   Visualizations: Distribution plots, feature importance, ROC curves
    (in Plots.zip)
-   Inference App: Lightweight app for prediction using trained models

## Installation

``` bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

Install dependencies (if requirements.txt is unavailable):

``` bash
pip install pandas numpy scikit-learn joblib torch ctgan flask
```

## Usage

### Run the notebook:

``` bash
jupyter notebook "Final FraudGen.ipynb"
```

### Launch the app:

``` bash
python app.py
```

## Results

Evaluation outputs are available in:

-   tstr_metrics.csv -- TSTR performance
-   validation_summary.csv -- Model validation results
-   Plots.zip -- Visualizations
-   Reports.zip -- Detailed evaluation reports

## Demo

Fraudgen demo 1.mp4 showcases the project flow and performance.

## Contributing

Contributions are welcome. Feel free to open issues or submit pull
requests.

## License

This project is licensed under the MIT License.
