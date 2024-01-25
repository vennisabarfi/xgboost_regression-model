
## XGBoost Regression for Continuous Data Prediction
xgboost regression model that models California housing data

## Results
Training score:  0.9419756057319272
CV mean score:  0.8302532469006225
MSE:  0.21333237170724623
RMSE:  0.10666618585362311

### Overview

This GitHub repository contains Python code that utilizes the XGBoost (Extreme Gradient Boosting) algorithm for predicting continuous data. The code specifically focuses on applications such as predicting heights, weights, temperatures, or any other continuous variable. The project uses the XGBRegressor from the XGBoost library, which is a powerful and efficient implementation of gradient boosting.

### Features

1. **Dataset:** The code uses the California Housing dataset, which is a well-known dataset for regression tasks. The dataset is loaded using scikit-learn's `fetch_california_housing` function.

2. **Data Splitting:** The dataset is split into training and testing sets using the `train_test_split` function from scikit-learn, with 85% of the data used for training and 15% for testing.

3. **XGBoost Model:** The XGBoost regression model is instantiated with adjustable parameters. The model is trained on the training data using the `fit` method.

4. **Training Evaluation:** The training score is calculated to assess the model's performance on the training data. A higher training score indicates a better fit.

5. **Cross-Validation:** K-fold cross-validation is performed using the `cross_val_score` function, providing an average training score across different subsets of the training data. This helps in evaluating the model's generalization performance.

6. **Test Data Prediction:** The trained model is used to predict the target variable on the test data.

7. **Accuracy Metrics:** Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) are calculated to quantify the difference between the predicted and actual values on the test set.

8. **Visualization:** The script includes a simple visualization using Matplotlib to compare the original and predicted values on the test set.

### Usage

To use this code, simply clone the repository and run the script. You may customize the dataset, adjust model parameters, and explore different evaluation metrics as needed.

Feel free to contribute, suggest improvements, or adapt the code for your specific regression tasks. Your feedback is highly appreciated!
