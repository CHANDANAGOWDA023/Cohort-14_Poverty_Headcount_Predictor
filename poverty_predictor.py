import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize
import os
import joblib

# Set aesthetic style for plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_and_clean_data(file_path):
    """Loads the dataset and performs basic cleaning."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()
    
    print("Dataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Handle missing values - for this project, we'll drop rows with missing values if any
    # as indicators for specific districts are critical.
    df = df.dropna()
    
    return df

def handle_outliers(df, column):
    """Detects and handles outliers using IQR and Winsorization."""
    print(f"\nHandling outliers for: {column}")
    
    # Visualization before handling
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df[column])
    plt.title(f"{column} Before Outlier Handling")
    
    # Calculate IQR
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f"IQR: {IQR}")
    print(f"Bounds: [{lower_bound}, {upper_bound}]")
    
    # Winsorization: capping values at 5th and 95th percentiles or custom bounds
    # Here we cap at the IQR bounds to be precise with the requirement
    df[column + '_Original'] = df[column] # Keep original for comparison
    
    # Using scipy winsorize for a fixed proportion or manual clip for IQR bounds
    # The requirement specifically mentions IQR method and Winsorization.
    # Usually, Winsorization uses percentiles, but we can clip at IQR bounds as a form of it.
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df[column])
    plt.title(f"{column} After Winsorization")
    plt.tight_layout()
    plt.savefig('outlier_handling_comparison.png')
    plt.show()
    
    return df

def perform_eda(df):
    """Performs Exploratory Data Analysis with visualizations."""
    print("\nPerforming EDA...")
    
    # 1. Correlation Heatmap
    # Filter only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Heatmap of Poverty Indicators")
    plt.savefig('correlation_heatmap.png')
    plt.show()
    
    # 2. Distribution Plot for Target Variable
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Headcount_Ratio (H)'], kde=True, color='teal')
    plt.title("Distribution of Headcount Ratio (H)")
    plt.savefig('headcount_distribution.png')
    plt.show()
    
    # 3. Pairplot for features vs target
    features = ['Average Household Size', 'Electricity Access', 'Average Annual Income']
    target = 'Headcount_Ratio (H)'
    
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(features):
        plt.subplot(1, 3, i+1)
        sns.scatterplot(data=df, x=col, y=target)
        plt.title(f"{col} vs {target}")
    plt.tight_layout()
    plt.savefig('feature_relationships.png')
    plt.show()

def train_and_evaluate_models(df):
    """Splits data, trains multiple models, and evaluates them."""
    features = ['Average Household Size', 'Electricity Access', 'Average Annual Income']
    target = 'Headcount_Ratio (H)'
    
    X = df[features]
    y = df[target]
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    best_model_name = ""
    best_r2 = -float('inf')
    best_pipeline = None
    
    print("\n--- Model Training & Evaluation ---")
    
    for name, model in models.items():
        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2,
            "Predictions": y_pred
        }
        
        print(f"\nModel: {name}")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipeline

    return results, best_model_name, best_pipeline, X_test, y_test, features

def visualize_results(results, best_model_name, best_pipeline, X_test, y_test, features):
    """Visualizes model performance and importance."""
    
    # 1. Actual vs Predicted for Best Model
    y_pred = results[best_model_name]["Predictions"]
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='b')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Headcount Ratio (H)')
    plt.ylabel('Predicted Headcount Ratio (H)')
    plt.title(f'Actual vs Predicted - {best_model_name}')
    plt.savefig('actual_vs_predicted.png')
    plt.show()
    
    # 2. Feature Importance for tree-based models
    if best_model_name in ["Random Forest Regressor", "Gradient Boosting Regressor"]:
        importance = best_pipeline.named_steps['regressor'].feature_importances_
        feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importance})
        feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feat_imp_df, x='Importance', y='Feature', palette='viridis')
        plt.title(f"Feature Importance ({best_model_name})")
        plt.savefig('feature_importance.png')
        plt.show()
        
    # Interpretation
    print("\n--- Interpretation of Model Performance ---")
    best_metrics = results[best_model_name]
    print(f"The best performing model is {best_model_name} with an R² score of {best_metrics['R2']:.4f}.")
    print(f"This indicates that approximately {best_metrics['R2']*100:.2f}% of the variance in ")
    print("Headcount Ratio can be explained by the selected features.")
    print(f"The RMSE of {best_metrics['RMSE']:.4f} represents the average deviation of predictions from actual values.")

def main():
    # File Path
    dataset_path = 'integrated_poverty_dataset.csv'
    
    try:
        # Step 1: Data Cleaning
        df = load_and_clean_data(dataset_path)
        
        # Step 2: Handle Outliers in 'Average Annual Income'
        df = handle_outliers(df, 'Average Annual Income')
        
        # Step 3: EDA
        perform_eda(df)
        
        # Step 4: Model Training and Evaluation
        results, best_model_name, best_pipeline, X_test, y_test, features = train_and_evaluate_models(df)
        
        # Step 5: Visualizations
        visualize_results(results, best_model_name, best_pipeline, X_test, y_test, features)
        
        # Step 6: Export Model for Frontend
        model_filename = "best_poverty_model.pkl"
        joblib.dump(best_pipeline, model_filename)
        print(f"\nModel exported successfully as {model_filename}")
        
        print("\nMachine Learning Workflow Completed Successfully!")
        print("Visualizations saved as PNG files in the current directory.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
