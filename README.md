# Smart Prep: Intelligent Preprocessing and Model Discovery

A comprehensive machine learning automation platform that simplifies the model building process while maintaining high performance standards.

## Features

- **Data Analysis Mode**: Explore your data with interactive visualizations and insights
  - Categorical vs. numerical feature relationships
  - Correlation heatmaps
  - Outlier detection
  - Distribution analysis
  - Detailed data summary

- **AutoML Modeling Mode**: Automatically build and evaluate ML models
  - Automated feature engineering
  - Hyperparameter tuning
  - Model selection and comparison
  - Performance evaluation
  - Explainable AI with SHAP values
  - Model download and prediction capabilities

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/smart-prep.git
cd smart-prep
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install AutoClean for advanced preprocessing:
```bash
pip install AutoClean
```

## Usage

1. Start the application:
```bash
streamlit run main.py
```

2. Upload your dataset (CSV or Excel format)

3. Choose your application mode:
   - **Data Analysis**: For exploring and understanding your data
   - **AutoML Modeling**: For building predictive models automatically

4. Follow the interface to analyze data or train models.

## Project Structure

```
project/
├── data/                # Stored datasets and processed data
├── models/              # Trained models
├── outputs/             # Visualizations and results
├── src/
│   ├── __init__.py
│   ├── preprocessing.py # Data cleaning functionality
│   ├── modeling.py      # ML modeling functionality
│   ├── visualization.py # Visualization utilities
│   └── utils.py         # Shared utility functions
├── main.py              # Main Streamlit application
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

## Dependencies

- H2O.ai AutoML: For automated machine learning
- Streamlit: For the web interface
- Pandas/NumPy: For data manipulation
- Matplotlib/Seaborn: For visualizations
- SHAP: For model explainability

## H2O Flow Interface Guide

H2O Flow is a web-based interactive user interface for working with H2O's machine learning platform. This section explains the key components of the H2O Flow interface:

### Header Navigation Bar
- **Flow dropdown**: Manage flow notebooks, save, download, or load notebooks
- **Cell dropdown**: Manipulate individual cells within the notebook
- **Data dropdown**: Import data, parse files, and manage datasets
- **Model dropdown**: Build, compare, and manage models
- **Score dropdown**: Make predictions and evaluate model performance
- **Admin dropdown**: System administration and monitoring
- **Help dropdown**: Access documentation and support resources

### Main Interface Components
1. **Flow Control Bar**: Buttons for document management, cell navigation, and execution
2. **Routines Panel**: Shows available operations including:
   - Data operations (importFiles, getFrames, splitFrame, mergeFrames)
   - Model operations (getModels, runAutoML, buildModel, predict)
   - Each operation has an icon and description explaining its function

### Key H2O Flow Features
- **Interactive Notebooks**: Combine code, text, visualizations in one document
- **Point-and-Click Operations**: Use without writing code
- **Model Building**: Access to all H2O algorithms (GBM, Random Forest, Deep Learning, etc.)
- **AutoML**: Automated model training and tuning
- **Real-time Visualization**: View model performance and data insights
- **Export Capabilities**: Download models as POJOs/MOJOs for production deployment

### Accessing H2O Flow
- Start H2O server with `java -jar h2o.jar`
- Access Flow interface at http://localhost:54321
- Use the Assist Me button for guided operations

This interface integrates seamlessly with the rest of our application, providing an alternative way to analyze data and build models alongside our Streamlit interface.

## Authors

- Azlan
- Mutlib
- Ameer Mortaza
- Anas Raza

Ghulam Ishaq Khan Institute of Engineering Sciences and Technology

## License

This project is licensed under the MIT License - see the LICENSE file for details. 