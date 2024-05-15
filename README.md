# Heart-Disease-Prediction-using-Tree-Based-Classifiers

This script showcases the process of training and evaluating tree-based classifiers on a heart disease dataset. It covers key machine learning concepts such as data preprocessing, model training, hyperparameter tuning, and model evaluation.

#### 1. **Imports and Setup**:
   - The script imports essential libraries like `numpy`, `pandas`, and various models from `sklearn` for machine learning tasks.
   - Additionally, it imports the `XGBClassifier` from `xgboost` for implementing XGBoost model training.

#### 2. **Loading and Preprocessing Data**:
   - The heart disease dataset is loaded using `pandas` from a CSV file.
   - Categorical variables are one-hot encoded using `get_dummies()` to prepare the data for modeling.

#### 3. **Train-Validation Split**:
   - The dataset is split into training and validation sets using `train_test_split()` from `sklearn`.
   - 80% of the data is used for training, and the remaining 20% is reserved for validation.

#### 4. **Model Training and Evaluation**:
   - Decision Tree and Random Forest models are trained and evaluated with varying hyperparameters (`min_samples_split` and `max_depth`) to find the optimal configuration.
   - Accuracy scores are computed for both training and validation sets to assess model performance.
   - The best-performing Random Forest model is selected based on accuracy.

#### 5. **XGBoost Model Training**:
   - An XGBoost model is trained with hyperparameters like `n_estimators` and `learning_rate`.
   - Early stopping is implemented using the `early_stopping_rounds` parameter to prevent overfitting.

#### 6. **Model Evaluation**:
   - The trained models are evaluated on both the training and validation sets to assess their performance using accuracy scores.
#### 7. **Visualization**:
  - Model performance metrics such as accuracy scores for training and validation sets are visualized using matplotlib.
  - Line plots depict how model accuracy varies with different hyperparameters (e.g., min_samples_split, max_depth, 
     n_estimators).

- ***Plot of minimum sample split***
![Plot-of-minimum-sample-split](https://github.com/UMMY87/Heart-Disease-Prediction-using-Tree-Based-Classifiers/assets/117314436/dfcd692d-ff43-4826-8ae8-f46d9ebdc1f2)

 - ***Plot of maximum depth***

![Plot-of-maximum-depth](https://github.com/UMMY87/Heart-Disease-Prediction-using-Tree-Based-Classifiers/assets/117314436/aa4b4a7e-6932-4517-a612-97336a512180)

 - ***plot of accuracy vs n-estimators***
![plot-of-accuracy-vs-n-estimators](https://github.com/UMMY87/Heart-Disease-Prediction-using-Tree-Based-Classifiers/assets/117314436/66e77192-75cf-4419-b50a-9e9f7b045c57)
    
#### Summary:
   - This script provides a detailed walkthrough of using tree-based classifiers for heart disease prediction.
   - It emphasizes the importance of hyperparameter tuning and model evaluation techniques in machine learning model development.
   - Additionally, it showcases the usage of various models available in scikit-learn and XGBoost for classification tasks.

#### License:
   - This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
