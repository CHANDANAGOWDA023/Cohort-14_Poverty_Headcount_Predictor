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
