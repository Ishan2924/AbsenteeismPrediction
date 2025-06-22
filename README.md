# Employee Absenteeism Prediction

This project focuses on building a machine learning model to predict employee absenteeism, aiming to identify key factors contributing to absence and provide a predictive tool. By analyzing various demographic, social, and professional attributes, the project seeks to help organizations better understand and manage workforce attendance.

## Problem Statement

Absenteeism is a significant concern for businesses, impacting productivity, resource allocation, and overall operational efficiency. This project addresses the challenge of predicting employee absenteeism by developing a robust classification model that can forecast whether an employee is likely to be absent, based on a comprehensive dataset of employee characteristics and historical absence records.

## Dataset

The project utilizes an absenteeism dataset which includes:
* **`Absenteeism-data.csv`**: The raw, original dataset.
* **`Absenteeism-preprocessed.csv`**: The cleaned and preprocessed version of the dataset, ready for model training.

The dataset features include various attributes such as:
* Reason for absence
* Date
* Transportation expense
* Distance from residence to work
* Age, BMI, Children, Pet
* Education, Social drinker/smoker status
* And other relevant demographic and professional factors.

##  Key Tasks & Workflow

The project follows a standard machine learning pipeline:

1.  **Exploratory Data Analysis (EDA):** In-depth analysis of the dataset to understand distributions, relationships between features, and identify patterns related to absenteeism.

2.  **Data Preprocessing:**
    * Handling missing values.
    * Feature engineering (e.g., extracting information from dates).
    * Scaling numerical features to normalize their range.
    * Encoding categorical variables.
    * Addressing multicollinearity (if applicable).

3.  **Model Training & Selection:**
    * Focus on **Logistic Regression** as the primary modeling approach for its interpretability and effectiveness in binary classification tasks.
    * Splitting data into training and testing sets.

4.  **Model Evaluation:** Assessing model performance using appropriate metrics (e.g., accuracy, precision, recall, F1-score).

5.  **Model Persistence:** Saving the trained machine learning model and the preprocessing objects (e.g., scalers, encoders) to disk for future use in prediction.

## Modeling Approach

The core of the prediction system is built around **Logistic Regression**. This model was chosen for its balance of performance and interpretability, making it suitable for understanding the likelihood of absence.

The `model.py` script houses the pipeline and the trained Logistic Regression model, along with the preprocessing steps. This ensures that the same transformations applied during training are consistently applied during inference.

## Project Structure

```
AbsenteeismPrediction/
├── Absenteeism_EDA.ipynb        # Jupyter Notebook for detailed Exploratory Data Analysis.
├── Absenteeism_Prediction.ipynb # Jupyter Notebook for data preprocessing, model training, and evaluation.
├── Absenteeism-data.csv         # The raw, original dataset used in the project.
├── Absenteeism-preprocessed.csv # Cleaned and preprocessed dataset, ready for model consumption.
├── model.py                     # Python script containing the preprocessor and the trained Logistic Regression model.
├── requirements.txt             # Lists all Python package dependencies with their exact versions.
└── .gitignore                   # Specifies files/directories to be excluded from Git tracking (e.g., virtual environment).
```
## Setup and Usage

Follow these steps to set up the project locally and interact with the model:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Ishan2924/AbsenteeismPrediction.git](https://github.com/Ishan2924/AbsenteeismPrediction.git)
    cd AbsenteeismPrediction
    ```

2.  **Create and Activate Virtual Environment:**
    It is highly recommended to use a Python virtual environment to isolate project dependencies.
    * **For Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    * **For macOS / Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install Dependencies:**
    With your virtual environment activated, install all necessary Python packages:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Explore Notebooks (Optional):**
    You can open and run the Jupyter Notebooks to understand the data processing and model training steps:
    ```bash
    jupyter notebook
    ```
    (Then navigate to `Absenteeism_EDA.ipynb` and `Absenteeism_Prediction.ipynb`)

5.  **Using the Pre-trained Model:**
    The `model.py` file contains the logic to load and use the pre-trained model and preprocessor. You can import these components into another Python script or a new Jupyter Notebook to make predictions on new data.

    ```python
    # Example of how to use the model.py components
    import pandas as pd
    from model import AbsenteeismModel

    # Load the model
    # Ensure 'Absenteeism_model.pkl' and 'Absenteeism_preprocessor.pkl'
    # (or similar names if saved differently within model.py) are in the project root
    # Adjust paths if your model.py saves them differently or requires different loading.
    model = AbsenteeismModel('Absenteeism_model.pkl', 'Absenteeism_preprocessor.pkl')

    # Example new data (replace with actual new data)
    new_data = pd.DataFrame([{
        'Reason for Absence': 23, 'Date': '2018-01-01', 'Transportation Expense': 289,
        'Distance to Work': 36, 'Age': 33, 'Daily Work Load Average': 239.554,
        'Body Mass Index': 30, 'Education': 1, 'Children': 2, 'Pets': 1,
        'Absenteeism Time in Hours': 0 # This would be the target, set to 0 or left out for prediction
    }])

    # Predict
    prediction = model.predict(new_data)
    print(f"Prediction: {prediction}")
    ```
    *Note: Verify the exact names of the saved model and preprocessor files (e.g., `Absenteeism_model.pkl`, `Absenteeism_preprocessor.pkl`) as they are saved/loaded within `model.py`.*

## Challenges & Key Learnings

* **Data Preparation Complexity:** Dealing with diverse data types, outliers, and preparing features for machine learning models (e.g., correct scaling and encoding for Logistic Regression) was a foundational challenge.
* **Feature Engineering:** Identifying and creating meaningful features from raw data (e.g., extracting day of the week or month from dates) to improve model performance.
* **Model Interpretability:** Choosing Logistic Regression allowed for insights into which factors significantly influence absenteeism, which is valuable for business decision-making.
* **Ensuring Consistency:** The importance of saving and loading both the trained model and the exact preprocessing steps (`model.py`) to ensure consistent data transformation between training and inference.

## Future Enhancements

* **Deployment as a Web Application:** Develop a Flask or Streamlit application for an interactive, user-friendly interface to input data and receive predictions.
* **Advanced Modeling:** Explore other classification algorithms (e.g., Random Forest, Gradient Boosting, SVM) and ensemble methods to potentially improve prediction accuracy.
* **Time-Series Analysis:** Incorporate time-series techniques to analyze absenteeism patterns over time and improve forecasting capabilities.
* **More Granular Prediction:** Instead of just binary classification (absent/not absent), predict the `Absenteeism Time in Hours` (regression) if the business need arises.
* **Feature Importance Visualization:** Add visualizations to show the importance of each feature in the model's predictions.
* **Live Data Integration:** If applicable, integrate with real-time HR data systems for automated absenteeism prediction.

![output](https://github.com/user-attachments/assets/a743ca44-5f5b-418e-93d8-3ded841aaa2a)
