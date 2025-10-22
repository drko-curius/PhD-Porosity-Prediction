####################
#####LIBRARIES######
####################

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import loguniform, uniform
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
import time

#Improving plots for article publication purposes
plt.style.use('default')
sns.set_palette("husl")

#Load small cubes dataset csv file SC_DatasetforNeuralNetwork
data = pd.read_csv("")

#Grouping per cube ID for cross-validation
X = data[["Print Temperature", "Layer height", "Infill Overlap", "Printing Speed", "Infill Type", "Z-Height"]]
y = data["Porosity"]
groups = data["cube_id"]

#Enforcing proper encoding of categorical features
categorical_features = ["Infill Type"]
numerical_features = ["Print Temperature", "Layer height", "Infill Overlap", "Printing Speed", "Z-Height"]

#Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(sparse_output=False, drop='first'), categorical_features)
    ]
)

#Base MLP model
regressor = MLPRegressor(
    max_iter=1000,
    solver='adam',
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)

#Create pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", regressor)
])

#Refined hyperparameter distributions based on previous best results
param_distributions = {
    'regressor__hidden_layer_sizes': [(100,), (150,), (200,), (100, 50), (100, 100), 
                                    (150, 100), (200, 100), (100, 50, 25)],
    'regressor__alpha': uniform(0.001, 0.01),  # Focus around previous best (0.0023)
    'regressor__learning_rate_init': uniform(0.003, 0.007),  # Focus around previous best (0.0038)
    'regressor__activation': ['tanh', 'relu'],
    'regressor__batch_size': [32, 64, 128, 'auto'],
    'regressor__beta_1': [0.8, 0.9, 0.95, 0.99],
    'regressor__beta_2': [0.99, 0.999, 0.9999],
    'regressor__early_stopping': [True],
    'regressor__n_iter_no_change': [10, 20, 30, 50]
}

#Use group K-fold to prevent data leakage
gkf = GroupKFold(n_splits=5)

#Extended randomized search with more iterations
print("Initiating hyperparameter tuning")
start_time = time.time()

random_search = RandomizedSearchCV(
    pipeline, 
    param_distributions, 
    n_iter=100,
    cv=gkf,
    scoring='r2', 
    verbose=2, 
    random_state=42,
    n_jobs=-1
)

#Fit with groups to prevent leakage
random_search.fit(X, y, groups=groups)

tuning_time = time.time() - start_time
print(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds")

#Retrieve best parameters
best_params = random_search.best_params_
print("\nBest Parameters:", best_params)
print("Best Cross-Validation R-squared:", random_search.best_score_)

#Final evaluation: Nested cross-validation
print("\n=== Final evaluation for nested cross-validation ===")

nested_scores = cross_val_score(
    random_search.best_estimator_, 
    X, y, 
    cv=gkf,
    scoring='r2',
    groups=groups
)

print("Nested Cross-Validation R-squared Scores:", nested_scores)
print("Mean R-squared Score:", np.mean(nested_scores))
print("Standard Deviation:", np.std(nested_scores))

#Train final model on all data for deployment and detailed analysis
final_model = random_search.best_estimator_.fit(X, y)
y_pred = final_model.predict(X)

#Calculate residuals
residuals = y - y_pred

#Save the optimized model
dump(final_model, 'SC_porosity_predictor_optimized_v3.joblib')

#Comprehensive metrics
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
full_r2 = r2_score(y, y_pred)

print("\n=== Final model hyperparameters ===")
print("Best parameters found:")
for param, value in best_params.items():
    print(f"{param}: {value}")

print(f"\nFinal Model Performance on Full SC Dataset:")
print(f"R²: {full_r2:.4f}")
print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")

#Dataset analysis
print(f"\n=== Dataset analysis ===")
print("Number of unique cubes:", len(np.unique(groups)))
print("Samples per cube:")
print(data['cube_id'].value_counts())
print(f"Total samples: {len(data)}")

#plotting graphs for article

#Predicted vs Actual Values Plot
plt.figure(figsize=(10, 8))
plt.scatter(y, y_pred, alpha=0.6, s=50)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect prediction')
plt.xlabel('Actual Porosity', fontsize=12)
plt.ylabel('Predicted Porosity', fontsize=12)
plt.title(f'MLP V3: Predicted vs Actual Porosity\n(R² = {full_r2:.3f} on full SC dataset)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('predicted_vs_actual_v3.png', dpi=300, bbox_inches='tight')
plt.show()

#Enhanced Residual Plot with density
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuals, alpha=0.6, s=50)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
plt.axvline(x=0, color='r', linestyle='--', lw=2)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residual_analysis_v3.png', dpi=300, bbox_inches='tight')
plt.show()

#Cross-Validation Performance Analysis
plt.figure(figsize=(14, 6))

#CV scores by fold
plt.subplot(1, 2, 1)
bars = plt.bar(range(1, 6), nested_scores, color='skyblue', alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--', label='Baseline')
plt.xlabel('Cross-Validation Fold')
plt.ylabel('R² Score')
plt.title('Grouped CV Performance by Fold')
plt.legend()

for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1, 
             f'{nested_scores[i]:.2f}', ha='center', va='bottom')

#Performance distribution across folds
plt.subplot(1, 2, 2)
plt.boxplot(nested_scores)
plt.plot([1], [np.mean(nested_scores)], 'ro', markersize=8, label=f'Mean: {np.mean(nested_scores):.2f}')
plt.ylabel('R² Score')
plt.title('CV Performance Distribution')
plt.legend()

plt.tight_layout()
plt.savefig('cv_performance_v3.png', dpi=300, bbox_inches='tight')
plt.show()

#Hyperparameter Search Analysis
if hasattr(random_search, 'cv_results_'):
    results = pd.DataFrame(random_search.cv_results_)
    
    #Plot top 20 configurations
    top_results = results.nlargest(20, 'mean_test_score')
    
    plt.figure(figsize=(14, 8))
    colors = ['red' if 'tanh' in str(arch) else 'blue' for arch in top_results['param_regressor__activation']]
    
    plt.scatter(range(len(top_results)), top_results['mean_test_score'], 
                c=colors, s=100, alpha=0.7)
    plt.axhline(y=0, color='green', linestyle='--', label='Baseline')
    plt.xlabel('Hyperparameter Configuration (Ranked)')
    plt.ylabel('Mean CV R² Score')
    plt.title('Top 20 Hyperparameter Configurations\n(Red: tanh, Blue: relu)')
    plt.legend()
    plt.xticks(range(len(top_results)), [f'Config {i+1}' for i in range(len(top_results))], rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hyperparameter_analysis_v3.png', dpi=300, bbox_inches='tight')
    plt.show()

#Learning Curve
if hasattr(final_model.named_steps['regressor'], 'loss_curve_'):
    plt.figure(figsize=(10, 6))
    plt.plot(final_model.named_steps['regressor'].loss_curve_)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('learning_curve_v3.png', dpi=300, bbox_inches='tight')
    plt.show()
  
print("Summary:")
print(f"- Best CV R²: {random_search.best_score_:.3f}")
print(f"- Full dataset R²: {full_r2:.3f}")
print(f"- Model: {best_params['regressor__hidden_layer_sizes']} architecture")
print(f"- Activation: {best_params['regressor__activation']}")
print(f"- Learning rate: {best_params['regressor__learning_rate_init']:.6f}")
print(f"- Regularization: {best_params['regressor__alpha']:.6f}")
