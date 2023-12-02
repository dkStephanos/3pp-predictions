# NBA 3pt% Predictive Modeling

![image](https://github.com/dkStephanos/3pp-predictions/blob/main/data/title.png)

## Project Overview
The `3pp-predictions` repository is dedicated to exploring advanced machine learning techniques for predicting the end-of-season three-point shooting percentage (3P%) of NBA players. This project leverages early-season shooting statistics to build models that can forecast player performance with higher accuracy. By combining statistical analysis with machine learning, this project aims to provide a nuanced understanding of player performance trends.

## Getting Started

### Jupyter Notebooks
The repository contains Jupyter notebooks which are the main drivers of the analysis:

- **Preprocessing.ipynb**
  - **Purpose**: Extends the base dataset with advanced statistics, visualizes the raw data, and performs LASSO-based feature selection.
  - **How to Run**: Open the notebook in a Jupyter environment and execute the cells sequentially. Ensure all dependencies are installed.
  - **Key Outputs**: Extended dataset with new features, visualizations of data distribution and relationships, selected features for model training.

- **Prediction.ipynb**
  - **Purpose**: Trains a variety of models to compare performance. Includes evaluation of model effectiveness through relevant statistics and learning/validation/loss curves.
  - **How to Run**: Similar to the Preprocessing notebook, run the cells in sequence to train models and generate performance metrics.
  - **Key Outputs**: Trained models, performance metrics (MSE, R² scores, etc.), visual plots depicting model learning and validation.

### Installation
To set up your environment to run these notebooks, you will need Python installed along with the necessary libraries. You can install the dependencies using:

```bash
pip install -r requirements.txt
```
This command will install all the required Python packages as listed in the requirements.txt file.

### Notes and Acknowledgements
This project builds on my previous NBA-related work available on my GitHub profile.
Parts of the code and documentation were developed with AI assistance.
Statistical definitions and conceptual insights were referenced from NBA Stuffer.

### Contributing
Contributions to 3pp-predictions are welcome! Please submit in the form of pull-requests of the main branch.

© 2023 Koi Stephanos. All Rights Reserved.