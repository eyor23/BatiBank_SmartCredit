# BatiBank_SmartCredit
A robust credit scoring system developed for Bati Bank's Buy-Now-Pay-Later service, featuring advanced feature engineering, modular code, comprehensive testing, and an API for real-time predictions.

# BatiBank_SmartCredit: Bati Bank's Buy-Now-Pay-Later Credit Scoring System

![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![Libraries](https://img.shields.io/badge/libraries-pandas%2C%20scikit--learn%2C%20flask-brightgreen.svg)
![Status](https://img.shields.io/badge/status-in%20progress-yellow.svg)

## Overview

BatiBank_SmartCredit is a sophisticated credit scoring solution designed for Bati Bank's Buy-Now-Pay-Later (BNPL) service. This project leverages advanced data analytics and machine learning techniques to assess customer credit risk, predict fraud, and optimize loan parameters. The system aims to provide Bati Bank with a robust, scalable, and reliable tool for managing credit risk in its BNPL operations.

## Project Goals

* Develop a predictive model to assess customer credit risk and detect fraudulent transactions.
* Implement a REST API for real-time credit scoring and fraud prediction.
* Ensure code modularity, reusability, and maintainability.
* Implement comprehensive unit and integration testing.
* Establish a basic CI/CD pipeline for automated testing and deployment.

## Data

The dataset used in this project contains transactional data from an eCommerce platform. It includes features such as:

* `TransactionId`
* `AccountId`
* `CustomerId`
* `Amount`
* `Value`
* `ProductCategory`
* `TransactionStartTime`
* `FraudResult`

**Note:** Due to data size and potential sensitivity, the raw dataset is not included in this repository. Please download the dataset from https://www.kaggle.com/datasets/atwine/xente-challenge


## Project Structure
BatiBank_SmartCredit/
├── data/
│   └── transaction_data.csv (Add this file after download)
├── notebooks/
│   └── EDA.ipynb
├── src/
│   ├── utils.py
│   └── visualization.py
├── api/
│   └── app.py
├── tests/
│   └── test_*.py
├── Dockerfile
├── requirements.txt
└── README.md



## Initial Exploratory Data Analysis (EDA) Findings

* **Data Overview:** The dataset contains various features related to transactions.
* **Missing Values:** The `CountryCode` feature is entirely missing, requiring further investigation or removal.
* **Numerical Features:** `Amount` and `Value` are highly correlated. `FraudResult` shows moderate correlation with `Amount` and `Value`.
* **Categorical Features:** The `FraudResult` is heavily imbalanced, with a significant skew towards non-fraudulent transactions.
* **Time-Based Analysis:** Transaction patterns reveal peak hours in the late afternoon and increased activity in December.
* **Fraud Analysis:** Fraudulent transactions are concentrated in specific product categories, such as financial\_services and airtime.

## Next Steps

1.  **Handle Missing Values:** Determine and implement a strategy for the missing `CountryCode` data.
2.  **Feature Engineering:** Develop and implement feature engineering strategies based on the EDA findings, including:
    * Time-based features.
    * Interaction features.
    * Aggregate features per customer.
    * Features based on the RFMS model.
3.  **Modularization:** Continue to extract reusable code into the `src` directory.
4.  **Testing:** Begin writing unit and integration tests.

## Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    ```

2.  Navigate to the project directory:

    ```bash
    cd FinSight-BatiCredit
    ```

3.  Create and activate a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate      # On Windows
    ```

4.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

5.  Download the dataset from [Kaggle Dataset Link Here] and place it in the `data` directory.

## Usage

(To be updated as the project progresses)

## Contributing

(To be updated as needed)

## License

(To be updated as needed)

## Contact

[Your Email or LinkedIn Profile]