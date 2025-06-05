# DyFAIP: End-to-End Imputation and Prediction via Adaptive Frequency Learning for Irregular Time Series

DyFAIP (Dynamic Frequency-Aware Imputation and Prediction) is a novel deep dynamic memory neural network designed to jointly perform missing data imputation and downstream prediction in an end-to-end framework. It is especially well-suited for irregularly sampled time series, such as those found in environmental or healthcare datasets.


📘 How to Use DyFAIP on the Beijing Air Quality Multi-Site Dataset

1. Preprocess the Dataset

Open and run the Jupyter notebook:

* Beijing Air Quality Data preprocessing.ipynb

    Step 1: Unzip the dataset inside the data/ directory.

    Step 2: Run the notebook to generate the input files required for training.

    Output: Preprocessed data ready for training (stored in the specified output directory).


2. Train and Evaluate the Model

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

Make sure the following key libraries are installed:

    torch

    numpy

    pandas

    scikit-learn

    matplotlib

    seaborn

    tqdm


1*) How to run the script on Beijing Air Quality Multi-Site dataset:

  * Beijing Air Quality Data preprocessing.ipynb

    -Unzipp the dataset from the data folder
    
    -Output : Generate the input data 


2*) Run the model on the dataset:

   * Using training.py
     
      -input_path : output path for the generated data from data pre-processing script!
     
      -Output : give the results of both tasks (Imputation results and downstream task results)
