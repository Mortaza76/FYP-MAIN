import streamlit as st
from streamlit.components.v1 import html
import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
import shap
from PIL import Image
from textwrap import wrap
from matplotlib.gridspec import GridSpec
from pandas.api.types import is_numeric_dtype, is_object_dtype
from scipy.stats import percentileofscore
import math

# ==============================================
# CSS LOADING WITH ABSOLUTE PATH
# ==============================================
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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
    st.image("logo.png", use_container_width=True)
    st.title("Configuration")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your dataset (CSV/Excel)", type=["csv", "xls", "xlsx"])
    
    # App mode selection
    app_mode = st.radio("Application Mode", ["Data Analysis", "AutoML Modeling"])
    
    if uploaded_file is not None:
        # Load data
        @st.cache_data
        def load_data(data_file):
            file_extension = data_file.name.split('.')[-1].lower()
            if file_extension in ['csv']:
                return pd.read_csv(data_file)
            elif file_extension in ['xls', 'xlsx']:
                return pd.read_excel(data_file)
            else:
                st.warning("Unsupported file format")
                return None
                
        data_df = load_data(uploaded_file)
        data_cat = data_df.select_dtypes(include="object")
        data_num = data_df.select_dtypes(include="number")
        data_num_cols = data_num.columns
        
        if app_mode == "Data Analysis":
            selected_cat = st.selectbox("Categorical Attribute", list(data_cat))
            selected_num = st.selectbox("Numerical Attribute", list(data_num))
            target_attribute = st.selectbox("Target Attribute for heatmap & outliers", list(data_num_cols))
            dataframe_select = st.radio("Select dataframe view", ["Full", "Numerical", "Categorical"])
            toggle_heatmap = st.toggle("Show Full Heatmap")
            
        elif app_mode == "AutoML Modeling":
            # Advanced settings
            with st.expander("⚙️ Advanced Settings"):
                max_models = st.slider("Maximum models to train", 5, 50, 10)
                max_runtime = st.number_input("Max runtime per model (minutes)", 1, 120, 5)
                problem_type = st.radio("Problem type", ["Auto Detect", "Classification", "Regression"])
                target_col = st.text_input("Target column (leave blank for auto-detect)", "")
    
    st.markdown("---")
    st.markdown("""
    **Final Year Project**  
    *Machine Learning Automation*  
    Developed by Azlan, Mutlib, Ameer Mortaza, and Anas Raza  
    [Ghulam Ishaq Khan Institute of Engineering Sciences and Technology](https://www.giki.edu.pk)  
    """)

# ==============================================
# DATA ANALYSIS CLASS
# ==============================================
class CategoricalFeatureHandler:
    def __init__(self, dataset):
        self.df = dataset.copy()
        
    def create_categories_info(self, cat_feature, num_feature):
        df = self.df
        
        info_df = (
            df.groupby(cat_feature)
            .agg(
                Median=(num_feature, np.nanmedian),
                Mean=(num_feature, np.nanmean),
                RelMeanDiff=(
                    num_feature,
                    lambda x: (np.nanmean(x) - np.nanmedian(x)) / np.nanmedian(x) * 100
                    if np.nanmedian(x) > 0
                    else 0,
                ),
            )
            .add_prefix(f"{num_feature} ")
        )
         
        for measure in ("Median", "Mean"):
            non_nan_values = df.loc[~df[num_feature].isna(), num_feature]
            info_df[f"{num_feature} {measure}Pctl."] = [
                percentileofscore(non_nan_values, score)
                for score in info_df[f"{num_feature} {measure}"]
            ]

        info_df["Counts"] = df[cat_feature].value_counts()
        info_df["Counts Ratio"] = df[cat_feature].value_counts(normalize=True)
        self.info_df = info_df
        
        self._provide_consistent_cols_order()
        return self.info_df.copy()
    
    def _provide_consistent_cols_order(self):
        (
            self._median_name,
            self._mean_name,
            self._rel_mean_diff_name,
            self._median_pctl_name,
            self._mean_pctl_name,
            self._counts_name,
            self._counts_ratio_name,
        ) = self.info_df.columns

        self.info_df = self.info_df[
            [
                self._counts_name,
                self._counts_ratio_name,
                self._median_name,
                self._median_pctl_name,
                self._mean_name,
                self._mean_pctl_name,
                self._rel_mean_diff_name,
            ]
        ]

        self._n_categories_in = self.info_df.shape[0]
        self._n_stats_in = self.info_df.shape[1]
        self._stat_names_in = self.info_df.columns
        
    def categories_info_plot(self, cat_feature, num_feature, palette="mako_r"):
        self.create_categories_info(cat_feature, num_feature)

        fig_height = 8
        if self._n_categories_in > 5:
            fig_height += (self._n_categories_in - 5) * 0.5

        fig = plt.figure(figsize=(12, fig_height), tight_layout=True)
        
        plt.suptitle(
            f"{cat_feature} vs {self._counts_name} & {self._median_name} & {self._rel_mean_diff_name}"
        )
        gs = GridSpec(nrows=2, ncols=3, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])  # Counts.
        ax2 = fig.add_subplot(gs[0, 1])  # Median.
        ax3 = fig.add_subplot(gs[0, 2])  # Relative Mean Diff.
        ax4 = fig.add_subplot(gs[1, :])  # Descriptive Stats.

        for ax, stat_name in zip(
            (ax1, ax2, ax3),
            (self._counts_name, self._median_name, self._rel_mean_diff_name),
        ):
            self._plot_category_vs_stat_name(ax, stat_name)
            if not ax == ax1:
                plt.ylabel("")

        self._draw_descriptive_stats(ax4)
        sns.set_palette("deep")  # Default palette.
        return fig
   
    def _plot_category_vs_stat_name(self, ax, stat_name):
        """Plots a simple barplot (`category` vs `stat_name`) in the current axis."""
        info_df = self.info_df
        order = info_df.sort_values(stat_name, ascending=False).index
        plt.sca(ax)
        plt.yticks(rotation=30)
        sns.barplot(data=info_df, x=stat_name, y=info_df.index, order=order)

    def _draw_descriptive_stats(self, ax4):
        """Draws info from the `info_df` at the bottom of the figure."""
        plt.sca(ax4)
        plt.ylabel("Descriptive Statistics", fontsize=12, weight="bold")
        plt.xticks([])
        plt.yticks([])

        # Spaces between rows and cols. Default axis has [0, 1], [0, 1] range,
        # thus we divide 1 by number of necessary rows / columns.
        xspace = 1 / (self._n_stats_in + 1)  # +1 due to one for a category.
        yspace = 1 / (self._n_categories_in + 1 + 1)  # +2 due to wide header.

        xpos = xspace / 2
        ypos = 1 - yspace
        wrapper = lambda text, width: "\n".join(line for line in wrap(text, width))


        for header in np.r_[["Category"], self._stat_names_in]:
            header = wrapper(header, 15)  # Wrap headers longer than 15 characters.
            plt.text(xpos, ypos, header, ha="center", va="center", weight="bold")
            xpos += xspace

        pattern = "{};{};{:.2%};{:,.1f};{:.0f};{:,.1f};{:.0f};{:+.2f}"
        category_stats = [pattern.format(*row) for row in self.info_df.itertuples()]

        for i, cat_stats in enumerate(category_stats):
            ypos = 1 - (5 / 2 + i) * yspace
            plt.axhline(ypos + yspace / 2, color="black", linewidth=5)
            for j, cat_stat in enumerate(cat_stats.split(";")):
                xpos = (1 / 2 + j) * xspace
                plt.text(xpos, ypos, cat_stat, ha="center", va="center")

def calculate_stats(dataframe):
    result_df = pd.DataFrame(columns=['Attribute', 'Mean', 'Median', 'Rel Mn-Md Diff'])   
    for i, column in enumerate(dataframe.columns):
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            mean = dataframe[column].mean()
            median = dataframe[column].median()
            relative_difference = abs(mean - median) / (median)*100 if median > 0 else 0
            temp_df = pd.DataFrame(
                {'Attribute': column, 
                 'Mean': mean, 
                 'Median': median, 
                 'Rel Mn-Md Diff': relative_difference
                }, index =[i])
            if median >= 10:
                result_df = pd.concat([result_df, temp_df], ignore_index=True)
    return result_df

# ==============================================
# MAIN APP CONTENT
# ==============================================
if uploaded_file is not None:
    if app_mode == "Data Analysis":
        # Create tabs for data analysis
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Chart", "🗃 Data", ":thermometer: Heat Map", "🔢 Outliers", "📊 Histograms"])
        
        with tab2:
            tab2.caption(f"Correlations between Categorical attribute {selected_cat} and Numerical Atrribute {selected_num}")

        # Create and display category info plot
        category_handler1 = CategoricalFeatureHandler(data_df)
        category_handler1.create_categories_info(selected_cat, selected_num)
        fig = category_handler1.categories_info_plot(selected_cat, selected_num)
        tab1.pyplot(fig)

        # Data summary container
        container = tab2.container()
        colA, colB, colC = container.columns(3)
        colA.write('Number of Numerical Attributes:')
        colA.write(len(data_num.columns))
        colB.write('Number of Categorical Attributes:')
        colB.write(len(data_cat.columns))
        colC.write("Total number of records")
        colC.write(len(data_df))

        # Display selected dataframe view
        if dataframe_select == "Full":   
            tab2.write(data_df)
        elif dataframe_select == "Numerical":
            tab2.write(data_num)
        elif dataframe_select == "Categorical":
            tab2.write(data_cat)
            tab2.write("Categorical attributes details:")

        # Display categorical attributes details
        with tab2:  
            container1 = st.container()
            container1.markdown("**<h4 style='text-align: center; color: lightgray;'>Details of Categorical Attributes</h4>**", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            threshold = 0.10    
            for i, attrib in enumerate(data_cat.columns):
                category_summary = {}
                column_to_write = None
                if (i % 3) == 0:
                    column_to_write = col1
                elif (i % 3) == 1:
                    column_to_write = col2
                elif (i % 3) == 2:
                    column_to_write = col3
                
                column_to_write.divider()
                column_to_write.write(f"**{data_cat[attrib].name}**")

                value_counts = data_cat[attrib].value_counts()
                normalized_value_counts = data_cat[attrib].value_counts(normalize=True)

                for value, count in value_counts.items():
                    normalized_count = normalized_value_counts[value]
                    if normalized_count < threshold:
                        category_summary.setdefault('Miscellaneous', {'count': 0, 'normalized_count': 0, 'categories':0})
                        category_summary['Miscellaneous']['count'] += count
                        category_summary['Miscellaneous']['categories'] = 1+category_summary['Miscellaneous']['categories']
                        category_summary['Miscellaneous']['normalized_count'] += normalized_count
                    else:
                        column_to_write.write(f"{value}: {count} ({normalized_count:.2%})")

                if 'Miscellaneous' in category_summary:
                    if category_summary['Miscellaneous']['categories'] == 1:
                        column_to_write.write(f"& {category_summary['Miscellaneous']['categories']} category: {category_summary['Miscellaneous']['count']} "
                        f"({category_summary['Miscellaneous']['normalized_count']:.2%})")
                    elif category_summary['Miscellaneous']['categories'] > 1:
                        column_to_write.write(f"& {category_summary['Miscellaneous']['categories']} categories: {category_summary['Miscellaneous']['count']} "
                        f"({category_summary['Miscellaneous']['normalized_count']:.2%})")

        # Heatmap visualization
        corr = data_num.corr()
        triu_mask_full = np.triu(corr)
        high_corr_cols = corr.loc[corr[target_attribute]>0.6, target_attribute].index
        high_corr = data_num[high_corr_cols].corr()
        triu_mask = np.triu(high_corr)

        with tab3:
            if toggle_heatmap:
                st.header("Intercorrelation Matrix Heatmap - Complete")
                fig_hm, ax = plt.subplots(figsize=(10,10))
                plt.style.use('dark_background')
                sns.heatmap(corr, square=True, annot=False, mask=triu_mask_full, ax=ax)
                st.pyplot(fig_hm)
            else:
                st.header("Intercorrelation Matrix Heatmap - Salients")
                fig_hm, ax = plt.subplots(figsize=(10,10))
                plt.style.use('dark_background')
                sns.heatmap(high_corr, square=True, annot=True, linewidth=2, mask=triu_mask, cmap='mako', ax=ax)
                st.pyplot(fig_hm)

        # Outliers visualization
        data_num_colsx = data_num_cols.drop([col for col in ["Id", "SalePrice"] if col in data_num_cols], errors="ignore")
        stats = calculate_stats(data_num[data_num_colsx])
        features = list(stats[stats['Rel Mn-Md Diff'] > 5]['Attribute'])
        num_rows = math.ceil(len(features)/3)
        outliers = list(data_df[features].max()*0.8)

        with tab4:
            st.header(f"Outliers versus target: **{target_attribute}**")
            fig_outliers, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(12,8), tight_layout=True, sharey=True)
            for i, (feature, outlier) in enumerate(zip(features, outliers)):
                ax = axes[i] if axes.ndim == 1 else axes[i // 3, i % 3]
                sns.scatterplot(x=data_df[feature], y=data_df[target_attribute], color="navy", ax=ax)
                df = data_df.loc[data_df[feature]>outlier, [feature, target_attribute]]
                sns.scatterplot(data=df, x=feature, y=target_attribute, ax=ax, color="red", marker="X")
            st.pyplot(fig_outliers)

        # Histograms visualization
        with tab5:   
            plt.style.use('dark_background')
            fig_hist, ax = plt.subplots(figsize=(20, 20))
            data_num.hist(ax=ax, xlabelsize=10, ylabelsize=10, color='#D0E11C', bins=30)
            st.pyplot(fig_hist)
            st.write(data_num.describe().T)

    elif app_mode == "AutoML Modeling":
        # Initialize H2O
        with st.status("🔧 Initializing H2O Cluster...", expanded=True) as status:
            try:
                h2o.init(max_mem_size="2G", nthreads=-1)
                st.success("H2O cluster initialized successfully!")
                status.update(label="✅ H2O Cluster Ready!", state="complete")
            except Exception as e:
                st.error(f"Failed to initialize H2O: {e}")
                st.stop()
        
        # Data overview
        with st.status("📊 Loading and Analyzing Data...", expanded=True) as status:
            try:
                st.session_state['data'] = data_df
                
                # Data overview
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Data Preview")
                    st.dataframe(data_df.head(), use_container_width=True)
                
                with col2:
                    st.subheader("Data Summary")
                    st.write(f"📏 **Shape:** {data_df.shape[0]} rows, {data_df.shape[1]} columns")
                    st.write(f"🔢 **Numeric Columns:** {len(data_df.select_dtypes(include=['number']).columns)}")
                    st.write(f"🔤 **Categorical Columns:** {len(data_df.select_dtypes(include=['object']).columns)}")
                    st.write(f"📅 **Date Columns:** {len(data_df.select_dtypes(include=['datetime']).columns)}")
                
                if data_df.shape[0] < 50:
                    st.warning("⚠️ Very small dataset. Results may not generalize well.")
                
                status.update(label="✅ Data Loaded Successfully!", state="complete")
            except Exception as e:
                st.error(f"Failed to load data: {e}")
                st.stop()
        
        # Problem type detection
        if problem_type == "Auto Detect":
            if target_col:
                target = target_col
            else:
                target = data_df.columns[-1]
            
            if data_df[target].dtype == 'object' or data_df[target].nunique() <= 20:
                detected_problem_type = "Classification"
            else:
                detected_problem_type = "Regression"
        else:
            detected_problem_type = problem_type
            if not target_col:
                target = data_df.columns[-1]
            else:
                target = target_col
        
        st.info(f"🔹 **Auto-detected Target:** {target} | **Problem Type:** {detected_problem_type}")
        
        # Convert to H2OFrame
        with st.spinner("Converting data to H2O frame..."):
            h2o_data = h2o.H2OFrame(data_df)
            features = [col for col in h2o_data.columns if col != target]
        
        # Train/Test Split
        with st.spinner("Splitting data into train/test sets..."):
            train, test = h2o_data.split_frame(ratios=[0.8], seed=1234)
            st.success(f"Data split complete: Train ({train.nrow} rows) | Test ({test.nrow} rows)")
        
        # Run AutoML
        st.header("🚀 Training AutoML Models")
        
        if st.button("Start AutoML Training", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.status("🏃‍♂️ Running AutoML...", expanded=True) as status:
                try:
                    aml = H2OAutoML(
                        max_models=max_models,
                        seed=1,
                        max_runtime_secs=max_runtime*60,
                        verbosity="info",
                        balance_classes=True if detected_problem_type == "Classification" else False
                    )
                    
                    start_time = time.time()
                    aml.train(x=features, y=target, training_frame=train)
                    training_time = time.time() - start_time
                    
                    st.session_state['aml'] = aml
                    st.session_state['training_time'] = training_time
                    
                    status.update(label=f"✅ AutoML Training Completed in {training_time:.2f} seconds!", state="complete")
                    
                    # Display leaderboard
                    st.subheader("🏆 Model Leaderboard")
                    leaderboard = aml.leaderboard.as_data_frame()
                    st.dataframe(leaderboard.style.background_gradient(cmap='YlGnBu'), use_container_width=True)
                    
                    # Best model info
                    leader_model = aml.leader
                    st.success(f"🎉 Best Model: {leader_model.model_id}")
                    
                    # Feature Importance
                    if leader_model.algo != "stackedensemble":
                        try:
                            st.subheader("📊 Feature Importance")
                            varimp = leader_model.varimp(use_pandas=True)
                            varimp = varimp.sort_values("relative_importance", ascending=False)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.barplot(x="relative_importance", y="variable", data=varimp, palette="viridis", ax=ax)
                            ax.set_xlabel("Relative Importance")
                            ax.set_title("Feature Importance")
                            st.pyplot(fig)
                        except Exception as e:
                            st.warning(f"Couldn't plot feature importance: {e}")
                    else:
                        st.info("Feature importance not available for Stacked Ensemble models.")
                    
                    # Model Performance
                    st.subheader("📈 Model Performance")
                    perf = leader_model.model_performance(test_data=test)
                    
                    if detected_problem_type == "Classification":
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Accuracy", f"{perf.accuracy()[0][1]:.4f}")
                        with col2:
                            st.metric("AUC", f"{perf.auc():.4f}" if perf.auc() is not None else "N/A")
                        with col3:
                            st.metric("LogLoss", f"{perf.logloss():.4f}")
                        
                        try:
                            st.subheader("ROC Curve")
                            roc_fig = plt.figure()
                            perf.plot(type="roc")
                            st.pyplot(roc_fig)
                        except:
                            st.warning("Could not plot ROC curve.")
                    
                    elif detected_problem_type == "Regression":
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("RMSE", f"{perf.rmse():.4f}")
                        with col2:
                            st.metric("MAE", f"{perf.mae():.4f}")
                        with col3:
                            st.metric("R²", f"{perf.r2():.4f}")
                    
                    # SHAP Summary Plot - Fixed implementation
                    try:
                        if leader_model.algo in ["gbm", "xgboost", "drf"]:
                            st.subheader("🧠 SHAP Explanation")
                            with st.spinner("Calculating SHAP values..."):
                                # Create a new figure for SHAP plot
                                shap_fig = plt.figure()
                                
                                # Calculate SHAP values - using test data
                                test_sample = test[:100, :]  # Use a sample for faster computation
                                explainer = shap.TreeExplainer(leader_model)
                                shap_values = explainer.shap_values(test_sample)
                                
                                # Plot SHAP summary
                                shap.summary_plot(shap_values, test_sample.as_data_frame(), plot_type="bar", show=False)
                                st.pyplot(shap_fig)
                    except Exception as e:
                        st.warning(f"Could not generate SHAP plot: {e}")
                    
                    # Model Comparison
                    st.subheader("🔍 Model Comparison")
                    model_metrics = []
                    leaderboard_df = aml.leaderboard.as_data_frame()
                    
                    for model_id in leaderboard_df['model_id']:
                        model = h2o.get_model(model_id)
                        performance = model.model_performance(test_data=test)
                        
                        if detected_problem_type == "Classification":
                            model_metrics.append({
                                'Model': model_id,
                                'AUC': performance.auc(),
                                'Accuracy': performance.accuracy()[0][1],
                                'LogLoss': performance.logloss()
                            })
                        elif detected_problem_type == "Regression":
                            model_metrics.append({
                                'Model': model_id,
                                'RMSE': performance.rmse(),
                                'MAE': performance.mae(),
                                'R2': performance.r2()
                            })
                    
                    comparison_df = pd.DataFrame(model_metrics)
                    
                    if detected_problem_type == "Classification":
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                        sns.barplot(x='Model', y='AUC', data=comparison_df, ax=ax1, palette="viridis")
                        ax1.set_title('AUC Comparison')
                        ax1.tick_params(axis='x', rotation=45)
                        
                        sns.barplot(x='Model', y='Accuracy', data=comparison_df, ax=ax2, palette="viridis")
                        ax2.set_title('Accuracy Comparison')
                        ax2.tick_params(axis='x', rotation=45)
                        
                        st.pyplot(fig)
                    
                    elif detected_problem_type == "Regression":
                        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
                        sns.barplot(x='Model', y='RMSE', data=comparison_df, ax=ax1, palette="viridis")
                        ax1.set_title('RMSE Comparison')
                        ax1.tick_params(axis='x', rotation=45)
                        
                        sns.barplot(x='Model', y='MAE', data=comparison_df, ax=ax2, palette="viridis")
                        ax2.set_title('MAE Comparison')
                        ax2.tick_params(axis='x', rotation=45)
                        
                        sns.barplot(x='Model', y='R2', data=comparison_df, ax=ax3, palette="viridis")
                        ax3.set_title('R² Comparison')
                        ax3.tick_params(axis='x', rotation=45)
                        
                        st.pyplot(fig)
                    
                    # Save model
                    try:
                        os.makedirs("models", exist_ok=True)
                        model_path = h2o.save_model(model=leader_model, path="models", force=True)
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
        if 'aml' in st.session_state:
            st.header("🔮 Make Predictions")
            
            sample_data = data_df.drop(columns=[target]).iloc[:5].copy()
            edited_data = st.data_editor(sample_data, num_rows="dynamic", use_container_width=True)
            
            if st.button("Predict on Sample Data"):
                with st.spinner("Making predictions..."):
                    try:
                        h2o_new_data = h2o.H2OFrame(edited_data)
                        predictions = st.session_state['aml'].leader.predict(h2o_new_data)
                        predictions_df = predictions.as_data_frame()
                        
                        st.subheader("Prediction Results")
                        st.dataframe(predictions_df.style.background_gradient(cmap='Greens'), use_container_width=True)
                        
                        # Download predictions
                        csv = predictions_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predictions",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
        
        # Final cleanup
        st.markdown("---")
        if st.button("Shutdown H2O Cluster"):
            with st.spinner("Shutting down H2O cluster..."):
                h2o.shutdown(prompt=False)
                st.success("H2O cluster shutdown successfully!")

else:
    st.info("ℹ️ Please upload a dataset to get started")
    st.image("logo2.png", use_container_width=True)
    
    # Project documentation
    with st.expander("📚 Project Documentation"):
        st.markdown("""
        ## Final Year Project: Automated Machine Learning System
        
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
    <p>Final Year Project | Machine Learning Automation | © 2023</p>
</div>
""", unsafe_allow_html=True)

