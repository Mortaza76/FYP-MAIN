"""
Smart Prep: Intelligent Preprocessing and Model Discovery

Main application entry point that provides a unified interface for 
data preprocessing, analysis, and machine learning model training.
"""
import streamlit as st
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Import local modules
from src.preprocessing import load_data, clean_data
from src.modeling import (
    initialize_h2o, shutdown_h2o, train_automl_models, 
    evaluate_model, get_feature_importance, save_model, 
    make_predictions, generate_shap_plot
)
from src.visualization import (
    CategoricalFeatureHandler, calculate_stats,
    generate_correlation_heatmap, generate_outliers_plot,
    generate_histograms, generate_model_comparison_plot
)
from src.utils import (
    load_css, save_figure, display_dataframe_info,
    create_download_link, timing_decorator, format_metrics_for_display
)

# App configuration
st.set_page_config(
    page_title="Smart Prep | ML Automation Platform",
    page_icon="🚀",
    layout="wide"
)

# Load CSS
load_css()

# ==============================================
# APP HEADER
# ==============================================
st.title("🚀 SMART PREP : INTELLIGENT PREPROCESSING AND MODEL DISCOVERY")
st.markdown("""
    <div class="highlighted-box">
    This application combines automated machine learning with comprehensive data analysis capabilities. 
    Upload your dataset to explore detailed insights or build predictive models using H2O's AutoML.
    </div>
    """, unsafe_allow_html=True)

# ==============================================
# SIDEBAR CONFIGURATION
# ==============================================
with st.sidebar:
    # Check if logo.png exists
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    else:
        st.title("Configuration")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your dataset (CSV/Excel)", type=["csv", "xls", "xlsx"])
    
    # App mode selection
    app_mode = st.radio("Application Mode", ["Data Analysis", "AutoML Modeling"])
    
    if uploaded_file is not None:
        # Load data
        data_df = load_data(uploaded_file)
        
        try:
            data_cat = data_df.select_dtypes(include="object")
            data_num = data_df.select_dtypes(include="number")
            data_num_cols = data_num.columns
            
            if app_mode == "Data Analysis":
                # Show categorical and numerical selection if available
                if not data_cat.empty:
                    selected_cat = st.selectbox("Categorical Attribute", 
                                        options=list(data_cat.columns), 
                                        index=0 if list(data_cat.columns) else None)
                
                if not data_num.empty:
                    selected_num = st.selectbox("Numerical Attribute", 
                                        options=list(data_num.columns),
                                        index=0 if list(data_num.columns) else None)
                    
                    target_attribute = st.selectbox("Target Attribute for heatmap & outliers", 
                                           options=list(data_num_cols),
                                           index=0 if list(data_num_cols) else None)
                
                dataframe_select = st.radio("Select dataframe view", ["Full", "Numerical", "Categorical"])
                toggle_heatmap = st.toggle("Show Full Heatmap")
                
            elif app_mode == "AutoML Modeling":
                # Advanced settings
                with st.expander("⚙️ Advanced Settings"):
                    max_models = st.slider("Maximum models to train", 5, 50, 10)
                    max_runtime = st.number_input("Max runtime per model (minutes)", 1, 120, 5)
                    problem_type = st.radio("Problem type", ["Auto Detect", "Classification", "Regression"])
                    target_col = st.text_input("Target column (leave blank for auto-detect)", "")
                    
                # Preprocessing options
                with st.expander("🧹 Preprocessing Options"):
                    preprocess_data = st.checkbox("Apply preprocessing", value=True)
                    if preprocess_data:
                        missing_num = st.selectbox("Handle missing numeric values", 
                                                  ["mean", "median", "mode", "drop"])
                        missing_categ = st.selectbox("Handle missing categorical values", 
                                                    ["mode", "drop", "most_frequent"])
                        handle_outliers = st.selectbox("Handle outliers", 
                                                      ["drop", "clip", "none"])
        except Exception as e:
            st.sidebar.error(f"Error analyzing dataset: {e}")
    
    st.markdown("---")
    st.markdown("""
    **Machine Learning Automation**  
    Final Year Project
    
    Developed by Azlan, Mutlib, Ameer Mortaza, and Anas Raza  
    [Ghulam Ishaq Khan Institute of Engineering Sciences and Technology](https://www.giki.edu.pk)  
    """)

# ==============================================
# MAIN APP CONTENT
# ==============================================
if uploaded_file is not None:
    # First time load, preprocess the data
    if 'preprocessed' not in st.session_state:
        with st.spinner("Processing dataset..."):
            # Apply preprocessing if in AutoML mode and preprocessing is enabled
            if app_mode == "AutoML Modeling" and preprocess_data:
                original_df, cleaned_df = clean_data(
                    uploaded_file,
                    save_path="data/cleaned_data.csv",
                    missing_num=missing_num if 'missing_num' in locals() else "mean",
                    missing_categ=missing_categ if 'missing_categ' in locals() else "mode",
                    outliers=handle_outliers if 'handle_outliers' in locals() else "drop"
                )
                st.session_state['original_df'] = original_df
                st.session_state['cleaned_df'] = cleaned_df
            else:
                # Just load the data for analysis
                st.session_state['original_df'] = data_df
                st.session_state['cleaned_df'] = data_df.copy()
            
            st.session_state['preprocessed'] = True
    
    # Use the active dataframe based on the mode
    df = st.session_state['cleaned_df'] if app_mode == "AutoML Modeling" else st.session_state['original_df']
    
    if app_mode == "Data Analysis":
        # Create tabs for data analysis
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Chart", "🗃 Data", "🔥 Heat Map", "🔢 Outliers", "📊 Histograms"])
        
        # Display categorical vs numerical chart
        with tab1:
            if all(var in locals() for var in ['selected_cat', 'selected_num']) and selected_cat is not None and selected_num is not None:
                try:
                    category_handler = CategoricalFeatureHandler(df)
                    fig = category_handler.categories_info_plot(selected_cat, selected_num)
                    tab1.pyplot(fig)
                except Exception as e:
                    tab1.error(f"Error creating chart: {e}")
            else:
                tab1.info("Please select both categorical and numerical attributes to visualize.")
        
        # Display data information
        with tab2:
            # Data summary
            container = st.container()
            cols = container.columns(3)
            
            # Display dataframe info
            num_cols = len(df.select_dtypes(include="number").columns)
            cat_cols = len(df.select_dtypes(include="object").columns)
            
            cols[0].metric("Numerical Attributes", num_cols)
            cols[1].metric("Categorical Attributes", cat_cols)
            cols[2].metric("Total Records", len(df))
            
            # Display selected dataframe view
            if dataframe_select == "Full":   
                st.dataframe(df, use_container_width=True)
            elif dataframe_select == "Numerical":
                st.dataframe(df.select_dtypes(include="number"), use_container_width=True)
            elif dataframe_select == "Categorical":
                st.dataframe(df.select_dtypes(include="object"), use_container_width=True)
                st.write("Categorical attributes details:")
                
                # Display categorical attributes details
                container = st.container()
                cols = container.columns(3)
                
                threshold = 0.10
                col_index = 0
                
                # Loop through categorical columns
                cat_cols = df.select_dtypes(include="object").columns
                for attrib in cat_cols:
                    category_summary = {}
                    col = cols[col_index % 3]
                    col_index += 1
                    
                    col.divider()
                    col.write(f"**{attrib}**")
                    
                    value_counts = df[attrib].value_counts()
                    normalized_value_counts = df[attrib].value_counts(normalize=True)
                    
                    for value, count in value_counts.items():
                        normalized_count = normalized_value_counts[value]
                        if normalized_count < threshold:
                            category_summary.setdefault('Miscellaneous', {'count': 0, 'normalized_count': 0, 'categories': 0})
                            category_summary['Miscellaneous']['count'] += count
                            category_summary['Miscellaneous']['categories'] += 1
                            category_summary['Miscellaneous']['normalized_count'] += normalized_count
                        else:
                            col.write(f"{value}: {count} ({normalized_count:.2%})")
                    
                    # Display miscellaneous categories
                    if 'Miscellaneous' in category_summary:
                        misc = category_summary['Miscellaneous']
                        col.write(f"& {misc['categories']} {'category' if misc['categories'] == 1 else 'categories'}: "
                                f"{misc['count']} ({misc['normalized_count']:.2%})")
        
        # Display correlation heatmap
        with tab3:
            if 'target_attribute' in locals() and target_attribute is not None:
                try:
                    fig = generate_correlation_heatmap(df, target_attribute, toggle_heatmap)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating heatmap: {e}")
            else:
                st.info("Please select a target attribute to visualize correlations.")
        
        # Display outliers visualization
        with tab4:
            if 'target_attribute' in locals() and target_attribute is not None:
                try:
                    st.header(f"Outliers versus target: **{target_attribute}**")
                    fig = generate_outliers_plot(df, target_attribute)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating outliers plot: {e}")
            else:
                st.info("Please select a target attribute to visualize outliers.")
        
        # Display histograms
        with tab5:
            try:
                fig = generate_histograms(df)
                st.pyplot(fig)
                
                # Show numeric stats
                st.write(df.select_dtypes(include="number").describe().T)
            except Exception as e:
                st.error(f"Error generating histograms: {e}")
    
    elif app_mode == "AutoML Modeling":
        # Initialize H2O
        if 'h2o_initialized' not in st.session_state:
            with st.status("🔧 Initializing H2O Cluster...", expanded=True) as status:
                try:
                    if initialize_h2o(max_mem_size="2G", nthreads=-1):
                        st.success("H2O cluster initialized successfully!")
                        st.session_state['h2o_initialized'] = True
                        status.update(label="✅ H2O Cluster Ready!", state="complete")
                    else:
                        st.error("Failed to initialize H2O.")
                        st.stop()
                except Exception as e:
                    st.error(f"Failed to initialize H2O: {e}")
                    st.stop()
        
        # Data overview
        with st.status("📊 Loading and Analyzing Data...", expanded=True) as status:
            try:
                # Data overview
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Data Preview")
                    st.dataframe(df.head(), use_container_width=True)
                
                with col2:
                    st.subheader("Data Summary")
                    display_dataframe_info(df)
                
                if df.shape[0] < 50:
                    st.warning("⚠️ Very small dataset. Results may not generalize well.")
                
                status.update(label="✅ Data Loaded Successfully!", state="complete")
            except Exception as e:
                st.error(f"Failed to load data: {e}")
                st.stop()
        
        # Problem type detection
        if 'target_col' in locals() and 'problem_type' in locals():
            if problem_type == "Auto Detect":
                if target_col:
                    target = target_col
                else:
                    target = df.columns[-1]
                
                if df[target].dtype == 'object' or df[target].nunique() <= 20:
                    detected_problem_type = "classification"
                else:
                    detected_problem_type = "regression"
            else:
                detected_problem_type = problem_type.lower()
                if not target_col:
                    target = df.columns[-1]
                else:
                    target = target_col
            
            st.info(f"🔹 **Auto-detected Target:** {target} | **Problem Type:** {detected_problem_type}")
        
        # Run AutoML
        st.header("🚀 Training AutoML Models")
        
        if st.button("Start AutoML Training", type="primary"):
            if 'target' not in locals() or 'detected_problem_type' not in locals():
                st.error("Problem type or target column not detected.")
                st.stop()
                
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.status("🏃‍♂️ Running AutoML...", expanded=True) as status:
                try:
                    # Train models
                    automl_results = train_automl_models(
                        df, 
                        target_col=target,
                        problem_type=detected_problem_type,
                        max_models=max_models if 'max_models' in locals() else 10,
                        max_runtime_secs=max_runtime * 60 if 'max_runtime' in locals() else 300,
                        seed=1
                    )
                    
                    if automl_results is None:
                        st.error("AutoML training failed.")
                        st.stop()
                    
                    # Store results in session state
                    st.session_state['automl_results'] = automl_results
                    
                    # Update status
                    status.update(label=f"✅ AutoML Training Completed in {automl_results['training_time']:.2f} seconds!", state="complete")
                    
                    # Display leaderboard
                    st.subheader("🏆 Model Leaderboard")
                    leaderboard = automl_results["leaderboard"]
                    st.dataframe(leaderboard.style.background_gradient(cmap='YlGnBu'), use_container_width=True)
                    
                    # Best model info
                    leader_model = automl_results["leader_model"]
                    st.success(f"🎉 Best Model: {leader_model.model_id}")
                    
                    # Feature Importance
                    varimp = get_feature_importance(leader_model)
                    if varimp is not None:
                        st.subheader("📊 Feature Importance")
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        plt.style.use('default')
                        sns.barplot(x="relative_importance", y="variable", data=varimp.head(10), palette="viridis", ax=ax)
                        ax.set_xlabel("Relative Importance")
                        ax.set_title("Feature Importance")
                        st.pyplot(fig)
                        
                        # Save figure
                        save_figure(fig, "feature_importance.png")
                    
                    # Model Performance
                    st.subheader("📈 Model Performance")
                    metrics = evaluate_model(
                        leader_model, 
                        automl_results["test"], 
                        automl_results["problem_type"]
                    )
                    
                    # Display metrics
                    format_metrics_for_display(metrics, automl_results["problem_type"])
                    
                    # ROC Curve for classification
                    if automl_results["problem_type"] == "classification":
                        try:
                            st.subheader("ROC Curve")
                            perf = leader_model.model_performance(test_data=automl_results["test"])
                            
                            # Instead of using perf.plot(type="roc"), create our own ROC curve
                            # Get TPR and FPR values from the performance object
                            fpr = perf.fprs
                            tpr = perf.tprs
                            
                            # Create custom ROC curve
                            roc_fig, ax = plt.subplots(figsize=(8, 6))
                            ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {perf.auc():.4f}')
                            ax.plot([0, 1], [0, 1], 'k--', linewidth=1)  # Diagonal line
                            
                            # Configure the plot
                            ax.set_xlabel('False Positive Rate')
                            ax.set_ylabel('True Positive Rate')
                            ax.set_title('ROC Curve')
                            ax.legend(loc='lower right')
                            ax.grid(True, alpha=0.3)
                            
                            # Set axis limits
                            ax.set_xlim([0, 1])
                            ax.set_ylim([0, 1.05])
                            
                            # Display the figure in Streamlit
                            st.pyplot(roc_fig)
                            
                            # Save figure
                            save_figure(roc_fig, "roc_curve.png")
                        except Exception as e:
                            st.warning(f"Could not plot ROC curve: {e}")
                    
                    # SHAP Summary Plot
                    try:
                        if leader_model.algo in ["gbm", "xgboost", "drf"]:
                            st.subheader("🧠 SHAP Explanation")
                            with st.spinner("Calculating SHAP values..."):
                                # Generate SHAP plot
                                shap_fig = generate_shap_plot(leader_model, automl_results["test"])
                                if shap_fig is not None:
                                    st.pyplot(shap_fig)
                                    
                                    # Save figure
                                    save_figure(shap_fig, "shap_summary.png")
                                else:
                                    st.info("SHAP plot generation skipped - not available for this model type")
                    except Exception as e:
                        st.warning(f"Could not generate SHAP plot: {str(e)}")
                        import traceback
                        st.write(traceback.format_exc())  # Add detailed error for debugging
                    
                    # Model Comparison
                    st.subheader("🔍 Model Comparison")
                    try:
                        comparison_fig = generate_model_comparison_plot(
                            automl_results["aml"], 
                            automl_results["test"], 
                            automl_results["problem_type"]
                        )
                        st.pyplot(comparison_fig)
                        
                        # Save figure
                        save_figure(comparison_fig, "model_comparison.png")
                    except Exception as e:
                        st.warning(f"Could not generate model comparison: {e}")
                    
                    # Save model
                    try:
                        model_path = save_model(leader_model, path="models", force=True)
                        if model_path:
                            st.success(f"Model saved successfully at: {model_path}")
                            
                            with open(model_path, "rb") as f:
                                st.download_button(
                                    label="Download Model",
                                    data=f,
                                    file_name=f"automl_model_{leader_model.model_id}.bin",
                                    mime="application/octet-stream"
                                )
                    except Exception as e:
                        st.error(f"Could not save the model: {e}")
                    
                except Exception as e:
                    st.error(f"AutoML training failed: {e}")
                    st.stop()
        
        # Inference Section
        if 'automl_results' in st.session_state:
            st.header("🔮 Make Predictions")
            
            # Create a sample of the dataframe without the target
            prediction_df = df.drop(columns=[st.session_state['automl_results']["target_col"]]).iloc[:5].copy()
            
            # Allow user to edit the sample data
            edited_data = st.data_editor(prediction_df, num_rows="dynamic", use_container_width=True)
            
            if st.button("Predict on Sample Data"):
                with st.spinner("Making predictions..."):
                    try:
                        # Make predictions
                        predictions_df = make_predictions(
                            st.session_state['automl_results']["leader_model"], 
                            edited_data
                        )
                        
                        if predictions_df is not None:
                            st.subheader("Prediction Results")
                            st.dataframe(predictions_df.style.background_gradient(cmap='Greens'), use_container_width=True)
                            
                            # Download predictions
                            create_download_link(predictions_df, "predictions.csv", "Download Predictions")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
        
        # Final cleanup
        st.markdown("---")
        if st.button("Shutdown H2O Cluster"):
            with st.spinner("Shutting down H2O cluster..."):
                shutdown_h2o()
                if 'h2o_initialized' in st.session_state:
                    del st.session_state['h2o_initialized']
                st.success("H2O cluster shutdown successfully!")

else:
    st.info("ℹ️ Please upload a dataset to get started")
    
    # Display logo if available
    if os.path.exists("logo2.png"):
        st.image("logo2.png", use_container_width=True)
    
    # Project documentation
    with st.expander("📚 Project Documentation"):
        st.markdown("""
        ## Machine Learning Automation System
        
        ### Objective
        Develop an automated machine learning platform that simplifies the model building process while maintaining high performance standards.
        
        ### Features
        - Comprehensive data analysis and visualization
        - Automated data preprocessing and feature engineering
        - Model selection and hyperparameter tuning
        - Performance evaluation and comparison
        - Explainable AI with SHAP values
        - Model deployment capabilities
        
        ### Technologies Used
        - H2O.ai AutoML
        - Streamlit for UI
        - SHAP for explainability
        - Pandas, Matplotlib, Seaborn for data analysis
        
        ### How It Works
        1. Upload your dataset (CSV or Excel format)
        2. Choose between Data Analysis or AutoML Modeling mode
        3. Configure settings and run the analysis/training
        4. Explore results and download outputs
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>Machine Learning Automation | Final Year Project | © 2023</p>
</div>
""", unsafe_allow_html=True)

# Add a "created with" attribution
st.markdown("""
<div style="text-align: center; font-size: 0.8em; color: #888;">
    Created with ❤️ using Streamlit and H2O.ai
</div>
""", unsafe_allow_html=True) 