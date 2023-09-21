#!/usr/bin/env python
# coding: utf-8

# # question 01
Gradient Boosting Regression, often referred to simply as Gradient Boosting or GBM, is an ensemble machine learning technique used for regression tasks. It builds an ensemble of decision trees in a sequential manner, where each tree corrects the errors made by the previous ones. The final prediction is a weighted sum of the predictions from all trees.

Here's how Gradient Boosting Regression works:

1. **Initialize with a Constant**:
   - The algorithm starts by making a simple prediction for all samples. This initial prediction is usually the mean (or median) of the target variable for the training set.

2. **Calculate Residuals**:
   - For each sample, calculate the difference between the actual target value and the current prediction. These differences are known as residuals.

3. **Train a Weak Learner (Decision Tree)**:
   - Train a weak learner (often a decision tree with limited depth) to predict the residuals. The weak learner focuses on capturing the patterns in the residuals.

4. **Update Predictions**:
   - Update the predictions by adding a fraction of the predictions from the new weak learner. The fraction is determined by a hyperparameter called the learning rate.

5. **Repeat Steps 2-4**:
   - Iterate through steps 2 to 4 for a predefined number of iterations or until a stopping criterion is met. Each new weak learner is trained to predict the residuals of the previous ensemble.

6. **Final Prediction**:
   - The final prediction is the sum of the initial constant prediction and the predictions from all weak learners.

Key Characteristics of Gradient Boosting Regression:

- **Sequential Training**: Like AdaBoost, Gradient Boosting trains weak learners sequentially, with each one focusing on the mistakes of its predecessors.

- **Gradient Descent Optimization**: The algorithm uses a gradient descent optimization technique to minimize the loss function, where the loss is defined as the difference between the actual target values and the current prediction.

- **Adaptive Learning**: Gradient Boosting adapts to the mistakes of the previous learners, emphasizing the importance of correctly predicting the residuals.

- **Combining Predictions**: The final prediction is a weighted sum of the predictions from all weak learners.

Gradient Boosting Regression is effective for regression tasks and is known for its ability to capture complex relationships in the data. It's important to tune hyperparameters like learning rate, maximum depth of trees, and the number of estimators to achieve the best performance. Popular implementations of Gradient Boosting include XGBoost, LightGBM, and CatBoost.
# # question 02

# In[ ]:


import numpy as np

# Generate synthetic dataset
np.random.seed(0)
X = np.random.rand(100, 1) * 10
y = 2*X.squeeze() + np.random.randn(100)

# Define a simple decision tree as weak learner
def decision_tree_predict(X, split_value):
    return np.where(X < split_value, 0, 1)

# Initialize constants
learning_rate = 0.1
n_estimators = 100

# Initialize predictions
predictions = np.full(y.shape, np.mean(y))

# Train weak learners
for _ in range(n_estimators):
    # Calculate residuals
    residuals = y - predictions

    # Train weak learner (decision tree)
    split_value = np.mean(X[residuals > 0])
    weak_learner_pred = decision_tree_predict(X, split_value)

    # Update predictions
    predictions += learning_rate * weak_learner_pred

# Calculate R-squared
total_variance = np.sum((y - np.mean(y))**2)
explained_variance = np.sum((predictions - np.mean(y))**2)
r_squared = 1 - (explained_variance / total_variance)

# Calculate Mean Squared Error
mse = np.mean((predictions - y)**2)

print(f"R-squared: {r_squared:.4f}")
print(f"Mean Squared Error: {mse:.4f}")


# # question 03

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

# Define the parameter grid to search
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 3, 4]
}

# Create a Gradient Boosting Regressor
gbm = GradientBoostingRegressor()

# Initialize GridSearchCV
grid_search = GridSearchCV(gbm, param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the model to the data
grid_search.fit(X, y)

# Get the best hyperparameters
best_learning_rate = grid_search.best_params_['learning_rate']
best_n_estimators = grid_search.best_params_['n_estimators']
best_max_depth = grid_search.best_params_['max_depth']

# Train the model with the best hyperparameters
best_gbm = GradientBoostingRegressor(learning_rate=best_learning_rate,
                                     n_estimators=best_n_estimators,
                                     max_depth=best_max_depth)
best_gbm.fit(X, y)

# Evaluate the model
mse = -grid_search.best_score_  # Negative because GridSearchCV uses negative MSE
r_squared = best_gbm.score(X, y)

print(f"Best Hyperparameters:")
print(f"Learning Rate: {best_learning_rate}")
print(f"Number of Estimators: {best_n_estimators}")
print(f"Max Depth: {best_max_depth}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r_squared:.4f}")


# # question 04
In the context of Gradient Boosting, a weak learner refers to a base model that performs slightly better than random chance on a given task. Specifically, it is a model that has the ability to learn from the data but is relatively simple and not highly expressive on its own. In practice, weak learners are often decision trees with limited depth.

Here are some characteristics of a weak learner:

1. **Limited Complexity**: A weak learner is deliberately kept simple. For example, in the case of decision trees, they are often shallow trees with only a few levels (sometimes referred to as "stumps").

2. **Slightly Better than Chance**: A weak learner is expected to perform slightly better than random guessing. In classification tasks, this means having an error rate that is less than 50%.

3. **Prone to Underfitting**: Since weak learners are intentionally kept simple, they may not capture complex relationships in the data. They are more prone to underfitting.

4. **Fast to Train**: Weak learners are typically quick to train because of their simplicity. This is advantageous in boosting algorithms where multiple weak learners are trained sequentially.

5. **Focused on Specific Patterns**: Weak learners are specialized in capturing specific patterns or features of the data. In Gradient Boosting, each weak learner focuses on correcting the errors of the previous ones.

The strength of Gradient Boosting lies in its ability to combine multiple weak learners in a sequential manner, with each one addressing a specific aspect of the data. By iteratively training weak learners and emphasizing the importance of samples that were previously misclassified, Gradient Boosting can build a strong, highly expressive model that captures complex relationships in the data.

Examples of weak learners include shallow decision trees, linear models, or even very simple neural networks with a small number of neurons. The choice of weak learner can have an impact on the performance and behavior of the Gradient Boosting model.
# # question 05
The intuition behind the Gradient Boosting algorithm can be summarized as follows:

1. **Sequential Error Correction**:
   - Gradient Boosting builds an ensemble of weak learners (e.g., decision trees) in a sequential manner. Each weak learner is trained to correct the errors of the previous ones.

2. **Focus on Residuals**:
   - At each iteration, the algorithm focuses on the residuals (the differences between the actual and predicted values) of the target variable. This allows the weak learner to concentrate on the samples that were not well-predicted by the ensemble so far.

3. **Gradient Descent Optimization**:
   - The algorithm uses a form of gradient descent optimization to minimize a loss function. It calculates the gradient of the loss with respect to the predictions, and then updates the predictions in the direction that minimizes the loss.

4. **Learning Rate**:
   - The learning rate hyperparameter controls the step size of the updates. A smaller learning rate makes the algorithm more conservative, while a larger learning rate allows for faster convergence but may lead to overshooting.

5. **Combination of Predictions**:
   - The final prediction is a weighted sum of the predictions from all weak learners. The weights are determined by both the learning rate and the performance of each weak learner.

6. **Complex Relationships**:
   - By sequentially training weak learners and placing more emphasis on the samples that were previously misclassified, the algorithm is able to capture complex relationships in the data.

7. **Avoiding Overfitting**:
   - By using weak learners and controlling the learning rate, Gradient Boosting tends to generalize well and is less prone to overfitting compared to individual, highly expressive models.

8. **Ensemble of Specialized Models**:
   - Each weak learner is specialized in capturing specific patterns or features of the data. The ensemble combines the strengths of these specialized models to create a more powerful, expressive model.

Overall, the intuition behind Gradient Boosting is to iteratively improve the model's predictions by focusing on the mistakes made in previous iterations. This adaptiveness and the ability to capture complex relationships make Gradient Boosting a powerful technique for regression and classification tasks.
# # question 06
The Gradient Boosting algorithm builds an ensemble of weak learners (typically decision trees) in a sequential manner. The process can be summarized as follows:

1. **Initialization**:
   - The algorithm starts by making an initial prediction for all samples. This initial prediction is usually the mean (or median) of the target variable for the training set.

2. **Calculate Residuals**:
   - For each sample, calculate the difference between the actual target value and the current prediction. These differences are known as residuals.

3. **Train a Weak Learner (Base Model)**:
   - Train a weak learner (often a decision tree with limited depth) on the training data using the residuals as the target variable. The weak learner focuses on capturing the patterns in the residuals.

4. **Update Predictions**:
   - Update the predictions by adding a fraction of the predictions from the new weak learner. The fraction is determined by a hyperparameter called the learning rate.

5. **Repeat Steps 2-4**:
   - Iterate through steps 2 to 4 for a predefined number of iterations or until a stopping criterion is met. Each new weak learner is trained to predict the residuals of the previous ensemble.

6. **Final Prediction**:
   - The final prediction is the sum of the initial constant prediction and the predictions from all weak learners.

Here's a bit more detail on each step:

- **Initialization**:
   - The initial prediction can be any constant value, but it's often chosen to be the mean (or median) of the target variable. This serves as a starting point for the iterative process.

- **Calculate Residuals**:
   - The residuals represent the errors made by the current ensemble on the training data. They are computed as the actual target values minus the current predictions.

- **Train a Weak Learner**:
   - The weak learner is trained using the training data, where the features are the original input features and the target variable is the residuals calculated in the previous step. The weak learner focuses on capturing patterns in the residuals.

- **Update Predictions**:
   - The predictions from the new weak learner are scaled by a factor determined by the learning rate. This factor controls the step size of the updates.

- **Repeat Steps 2-4**:
   - The process is repeated for a predefined number of iterations or until a stopping criterion (e.g., a threshold on the improvement in the loss function) is met.

- **Final Prediction**:
   - The final prediction is the sum of the initial constant prediction and the predictions from all weak learners. The ensemble of weak learners combines their individual strengths to create a more powerful model.

By sequentially training weak learners and emphasizing the importance of samples that were previously misclassified, Gradient Boosting builds a strong, highly expressive model that captures complex relationships in the data.
# # question 07
Constructing the mathematical intuition of the Gradient Boosting algorithm involves understanding the underlying principles and equations that govern the training process. Here are the key steps involved:

1. **Initialize Predictions**:
   - Start with an initial prediction for each sample in the training data. This initial prediction can be a simple value, often the mean (or median) of the target variable.

2. **Calculate Residuals**:
   - Calculate the residuals, which are the differences between the actual target values and the current predictions. These residuals represent the errors made by the current ensemble.

3. **Train a Weak Learner (Base Model)**:
   - Train a weak learner (typically a decision tree) on the training data, where the features are the original input features and the target variable is the residuals calculated in the previous step. The goal of this weak learner is to capture patterns in the residuals.

4. **Update Predictions**:
   - Scale the predictions from the weak learner by a factor determined by the learning rate. This factor controls the step size of the updates.

5. **Repeat Steps 2-4**:
   - Iterate through steps 2 to 4 for a predefined number of iterations or until a stopping criterion is met. Each new weak learner is trained to predict the residuals of the previous ensemble.

6. **Final Prediction**:
   - The final prediction is the sum of the initial constant prediction and the predictions from all weak learners. The ensemble of weak learners combines their individual strengths to create a more powerful model.

7. **Loss Function**:
   - Throughout the process, a loss function is used to quantify the difference between the actual target values and the predictions of the ensemble. Common loss functions for regression tasks include mean squared error (MSE) and absolute error.

8. **Gradient Descent Optimization**:
   - The algorithm employs a form of gradient descent optimization to minimize the loss function. This involves calculating the gradient of the loss with respect to the predictions, and then updating the predictions in the direction that minimizes the loss.

9. **Learning Rate**:
   - The learning rate hyperparameter controls the step size of the updates. A smaller learning rate makes the algorithm more conservative, while a larger learning rate allows for faster convergence but may lead to overshooting.

10. **Combining Predictions**:
   - The final prediction is a weighted sum of the predictions from all weak learners. The weights are determined by both the learning rate and the performance of each weak learner.

By understanding these steps and the underlying mathematics, one can develop a strong intuition for how Gradient Boosting works and how it builds a powerful ensemble model by sequentially correcting the errors of the previous learners. This understanding is crucial for effective hyperparameter tuning and model optimization.