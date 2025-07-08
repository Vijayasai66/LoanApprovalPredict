# LoanApprovalPredict
This project aims to predict whether a loan application will be approved or not based on various applicant details. The model is trained on historical loan data using multiple supervised machine learning algorithms.
## Project Structure
loan-approval-prediction/
├── Untitle.ipynb # Main Jupyter notebook with full pipeline
├── dataset.csv # Dataset used for training the models
├── README.md # Project documentation
└── requirements.txt # Dependencies for reproducing the project
## Problem Statement
Financial institutions face significant risk when granting loans. Predicting loan approval helps reduce default rates and improves decision-making. This project automates the decision process using machine learning.
## Dataset Overview
The dataset contains the following features:
- Applicant's income, co-applicant's income
- Loan amount and loan term
- Credit history
- Categorical features like gender, marital status, education, employment type, and property area
## Tools and Technologies Used
- **Language:** Python 3.x  
- **Notebook Environment:** Jupyter Notebook (SageMaker/Local)
- **Libraries:**  
  - `pandas`, `numpy` – Data manipulation  
  - `matplotlib`, `seaborn` – Data visualization  
  - `scikit-learn` – Model training and evaluation  
  - `imbalanced-learn` – Handling imbalanced data
  ## Project Pipeline
  ### 1. Data Preprocessing
- Handled missing values using mean, median, and mode
- Removed outliers (visualized using boxplots)
- Combined income features to form `Total_Income`
- Applied log transformation to normalize skewed distributions
  ### 2. Feature Engineering
- Created `Total_Income_log`, `LoanAmount_log`, etc.
- Removed redundant columns like `Loan_ID`
- Encoded categorical features using `LabelEncoder`
  ### 3. Data Splitting
  - Train-Test split with 75%-25% ratio using `train_test_split`
  ### 4. Model Training
  Trained and evaluated the following models:
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
  ### 5. Evaluation Metrics
- Accuracy Score
- Cross-validation
- Classification Report (Precision, Recall, F1-score)
  ### 6. Handling Imbalanced Data
  Used `RandomOverSampler` from `imbalanced-learn` to balance the `Loan_Status` label, retrained models on balanced data, and compared performance.

  
  ## Results and Observations
-Before balancing, models performed well but showed slight bias toward the majority class.

-After applying oversampling, models achieved more balanced precision and recall.

-Random Forest and Logistic Regression performed the best in terms of generalization.


