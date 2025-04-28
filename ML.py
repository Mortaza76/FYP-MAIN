import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

print("🔄 Script started...")

# 🔹 Initialize H2O
try:
    h2o.init(max_mem_size="2G", nthreads=-1)
except Exception as e:
    print(f"❌ Failed to initialize H2O: {e}")
    sys.exit(1)

# 🔹 Load dataset
DATA_FILE = "cleaned_test.csv"
if not os.path.exists(DATA_FILE):
    print(f"❌ Dataset file '{DATA_FILE}' not found!")
    h2o.shutdown(prompt=False)
    sys.exit(1)

data = pd.read_csv(DATA_FILE)

if data.shape[0] < 50:
    print(f"⚠️ Warning: Very small dataset ({data.shape[0]} rows). Results may not generalize well.")

# 🔹 Auto-detect target column (last column assumed)
target = data.columns[-1]
print(f"[INFO] Auto-detected target column: {target}")

# 🔹 Detect problem type
if data[target].dtype == 'object' or data[target].nunique() <= 20:
    problem_type = "classification"
else:
    problem_type = "regression"
print(f"[INFO] Detected problem type: {problem_type}")

# 🔹 Convert to H2OFrame
h2o_data = h2o.H2OFrame(data)

# 🔹 Set features
features = [col for col in h2o_data.columns if col != target]

# 🔹 Train/Test Split
train, test = h2o_data.split_frame(ratios=[0.8], seed=1234)

def display_model_comparison_plot(aml, test_data, problem_type):
    print("\n[INFO] Generating Model Comparison Plot...")
    
    # Initialize a list to hold model metrics
    model_metrics = []

    # Extract leaderboard from AutoML
    leaderboard_df = aml.leaderboard.as_data_frame()

    for model_id in leaderboard_df['model_id']:
        model = h2o.get_model(model_id)
        
        # Evaluate model performance on test data
        try:
            performance = model.model_performance(test_data=test_data)
            print(f"[INFO] Performance for model {model_id}: {performance}")
            
            if problem_type == "classification":
                auc = performance.auc()
                accuracy = performance.accuracy()[0][1]
                logloss = performance.logloss()
                model_metrics.append({
                    'model_id': model_id,
                    'AUC': auc,
                    'Accuracy': accuracy,
                    'LogLoss': logloss
                })
            elif problem_type == "regression":
                rmse = performance.rmse()
                mae = performance.mae()
                r2 = performance.r2()
                model_metrics.append({
                    'model_id': model_id,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2
                })

        except Exception as e:
            print(f"[WARNING] Failed to evaluate model performance for {model_id}: {e}")

    # Create a DataFrame for plotting
    model_comparison_df = pd.DataFrame(model_metrics)
    
    # Plot comparison based on classification or regression
    if problem_type == "classification":
        # Plot AUC and Accuracy for Classification Models
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # AUC Plot
        sns.barplot(x='model_id', y='AUC', data=model_comparison_df, ax=ax[0], palette="viridis")
        ax[0].set_title('AUC Comparison')
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha='right')
        ax[0].set_ylabel('AUC')
        
        # Accuracy Plot
        sns.barplot(x='model_id', y='Accuracy', data=model_comparison_df, ax=ax[1], palette="viridis")
        ax[1].set_title('Accuracy Comparison')
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha='right')
        ax[1].set_ylabel('Accuracy')

        plt.tight_layout()
        plt.show()
        
    elif problem_type == "regression":
        # Plot RMSE, MAE, and R2 for Regression Models
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        
        # RMSE Plot
        sns.barplot(x='model_id', y='RMSE', data=model_comparison_df, ax=ax[0], palette="viridis")
        ax[0].set_title('RMSE Comparison')
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha='right')
        ax[0].set_ylabel('RMSE')

        # MAE Plot
        sns.barplot(x='model_id', y='MAE', data=model_comparison_df, ax=ax[1], palette="viridis")
        ax[1].set_title('MAE Comparison')
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha='right')
        ax[1].set_ylabel('MAE')
        
        # R² Plot
        sns.barplot(x='model_id', y='R2', data=model_comparison_df, ax=ax[2], palette="viridis")
        ax[2].set_title('R2 Comparison')
        ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=45, ha='right')
        ax[2].set_ylabel('R²')
        plt.tight_layout()
        plt.show()
# 🔹 Run AutoML
print("[INFO] Starting AutoML...")
aml = H2OAutoML(
    max_models=10,
    seed=1,
    verbosity="info",
    balance_classes=True if problem_type == "classification" else False
)
aml.train(x=features, y=target, training_frame=train)

# 🔹 Leaderboard
print("\n[INFO] AutoML Leaderboard:")
print(aml.leaderboard.head(rows=10))

# 🔹 Best Model
leader_model = aml.leader
print(f"\n[INFO] Best model: {leader_model.model_id}")

# 🔹 Feature Importance
if leader_model.algo != "stackedensemble":
    try:
        varimp = leader_model.varimp(use_pandas=True)
        varimp = varimp.sort_values("relative_importance", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="relative_importance", y="variable", data=varimp, palette="viridis")
        plt.xlabel("Relative Importance")
        plt.title("Feature Importance")
        plt.tight_layout()

        os.makedirs("outputs", exist_ok=True)
        feature_importance_path = os.path.join("outputs", "feature_importance.png")
        plt.savefig(feature_importance_path)
        print(f"[INFO] Feature importance chart saved as '{feature_importance_path}'")
        plt.show()

    except Exception as e:
        print(f"[WARNING] Couldn't plot feature importance: {e}")
else:
    print("[INFO] Feature importance not available for Stacked Ensemble models.")

# 🔹 Model Performance
print("\n[INFO] Evaluating model on test data...")
perf = leader_model.model_performance(test_data=test)

if problem_type == "classification":
    accuracy = perf.accuracy()[0][1]
    print(f"✅ Accuracy: {accuracy:.4f}")

    auc = perf.auc()
    if auc is not None:
        print(f"✅ AUC: {auc:.4f}")
        try:
            perf.plot(type="roc")
        except:
            print("[WARNING] Could not plot ROC curve.")

    logloss = perf.logloss()
    print(f"✅ LogLoss: {logloss:.4f}")

elif problem_type == "regression":
    print(f"✅ RMSE: {perf.rmse():.4f}")
    print(f"✅ MAE: {perf.mae():.4f}")
    print(f"✅ R2: {perf.r2():.4f}")

# 🔹 Displaying explainability and model insights
# Call the function to generate feature importance, PDPs, and model performance
def display_auto_ml_explainability(aml, test_data):
    print("\n[INFO] Generating feature importance and model explainability for all models...")
    
    # Iterate over all models in the AutoML leaderboard
    for model_id in aml.leaderboard['model_id'].as_data_frame().iloc[:, 0]:
        model = h2o.get_model(model_id)
        print(f"\n[INFO] Generating explainability for model: {model_id}")
        
        # Generate and plot variable importance (common across many models)
        try:
            varimp = model.varimp(use_pandas=True)
            varimp = varimp.sort_values("relative_importance", ascending=False)
            plt.figure(figsize=(10, 6))
            sns.barplot(x="relative_importance", y="variable", data=varimp, palette="viridis")
            plt.xlabel("Relative Importance")
            plt.title(f"Feature Importance for {model_id}")
            plt.tight_layout()

            os.makedirs("outputs", exist_ok=True)
            feature_importance_path = os.path.join("outputs", f"{model_id}_feature_importance.png")
            plt.savefig(feature_importance_path)
            print(f"[INFO] Feature importance chart saved as '{feature_importance_path}'")
            plt.show()

        except Exception as e:
            print(f"[WARNING] Couldn't generate variable importance for {model_id}: {e}")
        
        # Evaluate model performance on test data
        try:
            performance = model.model_performance(test_data=test_data)
            print(f"[INFO] Model performance for {model_id}:\n{performance}")
        except Exception as e:
            print(f"[WARNING] Failed to evaluate model performance for {model_id}: {e}")

        # Generate Partial Dependence Plots (PDP) for important features
        features = model.features()  # This method gives the model's feature names
        for feature in features:
            if feature != model.response():
                try:
                    generate_pdp(model, feature, test_data)
                except Exception as e:
                    print(f"[WARNING] Failed to generate PDP for {feature} in {model_id}: {e}")

# Function to generate Partial Dependence Plot (PDP) for a given model and feature
def generate_pdp(model, feature_name, test_data):
    print(f"[INFO] Generating PDP for model {model.model_id} and feature {feature_name}")
    
    try:
        pdp = model.partial_dependence_plot(feature_name=feature_name, data=test_data)
        plt.figure(figsize=(10, 6))
        plt.plot(pdp['grid'], pdp['plot'])
        plt.title(f"PDP for {feature_name} in Model {model.model_id}")
        plt.xlabel(feature_name)
        plt.ylabel("Partial Dependence")
        plt.tight_layout()
        
        os.makedirs("outputs", exist_ok=True)
        pdp_path = os.path.join("outputs", f"{model.model_id}_PDP_{feature_name}.png")
        plt.savefig(pdp_path)
        print(f"[INFO] PDP chart saved as '{pdp_path}'")
        plt.show()
    except Exception as e:
        print(f"[WARNING] Failed to generate PDP for {model.model_id}: {e}")


# 🔹 Save Best Model
try:
    os.makedirs("models", exist_ok=True)
    model_path = h2o.save_model(model=leader_model, path="models", force=True)
    print(f"[INFO] Best model saved at: {model_path}")
except Exception as e:
    print(f"[ERROR] Could not save the model: {e}")
# 🔹 Explainable AI (SHAP summary plot)
# 🔹 Explainable AI (SHAP summary plot)
# 🔹 Explainable AI (SHAP summary plot)
try:
    print("\n[INFO] Generating Explainable AI (SHAP Summary Plot)...")
    
    if leader_model.algo in ["gbm", "xgboost", "drf"]:  # Check for supported models
        shap_values = leader_model.shap_values(test)  # Get SHAP values from test data
        
        # Plot SHAP summary plot
        shap_values = shap_values.as_data_frame()  # Convert to pandas dataframe
        shap_summary = shap_values.describe()  # You can use .describe() or .head() to view a summary of SHAP values
        
        print("[INFO] SHAP summary generated successfully.")
        
        # Visualize SHAP summary plot (optional)
        shap.summary_plot(shap_values.as_data_frame(), test)  # Visualize with SHAP library if needed
        
    else:
        print(f"[INFO] SHAP plots not supported for {leader_model.algo} models.")
        
except Exception as e:
    print(f"[WARNING] Could not generate SHAP plot: {e}")
display_model_comparison_plot(aml=aml, test_data=test, problem_type="classification")

# 🔹 Inference on New Data
try:
    print("\n[INFO] Running inference on new data...")
    new_data = pd.read_csv(DATA_FILE)  # Could also be different
    h2o_new_data = h2o.H2OFrame(new_data)

    predictions = leader_model.predict(h2o_new_data)

    predictions_df = predictions.as_data_frame()
    print("\n[INFO] Sample predictions:")
    print(predictions_df.head())

    output_preds_path = os.path.join("outputs", "sample_predictions.csv")
    predictions_df.to_csv(output_preds_path, index=False)
    print(f"[INFO] Predictions saved at '{output_preds_path}'")

except Exception as e:
    print(f"[WARNING] Skipped inference: {e}")

# 🔹 Shutdown H2O
h2o.shutdown(prompt=False)
print("\n✅ All tasks completed successfully!")
