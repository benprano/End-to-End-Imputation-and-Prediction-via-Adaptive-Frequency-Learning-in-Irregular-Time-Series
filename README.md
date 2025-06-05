# DyFAIP: End-to-End Imputation and Prediction via Adaptive Frequency Learning for Irregular Time Series

DyFAIP (Dynamic Frequency-Aware Imputation and Prediction) is a novel deep dynamic memory neural network designed to jointly perform missing data imputation and downstream prediction in an end-to-end framework. It is especially well-suited for irregularly sampled time series, such as those found in environmental or healthcare datasets.


### 📘 How to Use DyFAIP on the Beijing Air Quality Multi-Site Dataset

### 🧹 1. Preprocess the Dataset


Open and run the Jupyter notebook:

* Beijing Air Quality Data preprocessing.ipynb

    Step 1: Unzip the dataset inside the data/ directory.

    Step 2: Run the notebook to generate the input files required for training.

    Output: Preprocessed data ready for training (stored in the specified output directory).


### 🚀  2. Train and Evaluate the Model


Use the script:
training.py

    Required argument:

        --input_path: Path to the processed dataset generated from the preprocessing step.

    Output:

        Quantitative results for both imputation and prediction tasks, including RMSE, MAE, and performance metrics on the downstream task.

🔧 1. Environment Setup

We recommend using Conda:

conda create -n dyfaip python=3.9 -y
conda activate dyfaip
pip install -r requirements.txt

---

## 🌍 3. Make sure the following key libraries are installed:

* `torch`
* `numpy`
* `pandas`
* `scikit-learn`
* `matplotlib`
* `seaborn`
* `tqdm`

---


## 🧠 4. DyFAIP Model Overview

DyFAIP is built with the following components:

    Frequency-Aware Temporal : Learns both short- and long-range frequency patterns in irregular time series.

    Dynamic Memory Module: Aggregates context over time using adaptive memory gates.

    Joint Optimization: Simultaneously optimizes imputation loss and prediction loss.

**Supported tasks:**

* Time series **imputation**
* Time series **forecasting**
* Time series **classification**

**Evaluation metrics for both:**

    * **Imputation** (e.g., RMSE, MAE, Adj $R^2$)
    * **Downstream prediction** (e.g., AUPRC, F1-score)


## 📊 Results

Example performance on the Beijing Air Quality dataset:

| Setting    | RMSE | Adj $R^2$ |
| ---------- | ---- | --------- |
| MAR (20%)  | 0.06 | 0.89      |
| MNAR (50%) | 0.07 | 0.89      |

---

