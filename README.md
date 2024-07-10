# 📦 PCA and Ensemble Learning App

Our project explores Principal Component Analysis (PCA) for dimensionality reduction and various ensemble learning techniques to improve model performance, providing detailed insights from diverse datasets.

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ml-model-builder-template.streamlit.app/)

## Project Team

- Oussama Hbouz
- Mohamed Marzak
- Hanane Mdihen
- Douaa Khiri
- Hiba Benkattaba
- Houda Fattoumi

## Introduction

This Streamlit app allows users to perform Principal Component Analysis (PCA) and explore various ensemble learning techniques using predefined datasets or their own data.

## Features

- **Principal Component Analysis (PCA)**
  - Select numerical features for PCA
  - Visualize PCA results with scatter plots and scree plots
  - Display explained variance and cumulative variance

- **Ensemble Learning Methods**
  - Voting Classifier and Regressor
  - Bagging Classifier and Regressor
  - Boosting Classifier and Regressor
  - Stacking Classifier and Regressor

## How to Use

1. **Select or Upload Data**
   - Choose a predefined dataset from the dropdown menu or upload your own CSV file.
2. **Select Analysis Type**
   - Choose either PCA or Ensemble Learning.
3. **PCA Analysis**
   - Select numerical features and adjust the number of PCA components.
   - View the PCA results and visualizations.
4. **Ensemble Learning**
   - Select the task type (Classification or Regression).
   - Choose the ensemble method and adjust relevant parameters.
   - View model performance and visualizations.

## Installation

To run this app locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
2. Install the required packages:
     ```bash
     pip install -r requirements.txt
3. Run the Streamlit app:
    ```bash
    streamlit run app.py

## License
This project is licensed under the MIT License.