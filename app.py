import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import plotly.express as px
from sklearn.metrics import accuracy_score, r2_score

# --- Import All Models ---
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, LogisticRegression, Perceptron
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA, KernelPCA, FastICA, FactorAnalysis, NMF
from sklearn.manifold import TSNE
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

# --- Model & Explanation Dictionaries ---
MODELS = {
    "Supervised Learning": {
        "Regression": {
            "Linear Regression": LinearRegression(), "Ridge Regression": Ridge(random_state=42), "Lasso Regression": Lasso(random_state=42),
            "Elastic Net Regression": ElasticNet(random_state=42), "Polynomial Regression": Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())]),
            "Bayesian Linear Regression": BayesianRidge(), "Support Vector Regression (SVR)": SVR(), "Decision Tree Regression": DecisionTreeRegressor(random_state=42),
            "Random Forest Regression": RandomForestRegressor(random_state=42), "Gradient Boosting Regression (GBR)": GradientBoostingRegressor(random_state=42),
            "k-Nearest Neighbors Regression (kNN)": KNeighborsRegressor(), "XGBoost Regression": xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
            "LightGBM Regression": lgb.LGBMRegressor(random_state=42), "CatBoost Regression": cb.CatBoostRegressor(verbose=0, random_state=42),
        },
        "Classification": {
            "Logistic Regression": LogisticRegression(random_state=42), "k-Nearest Neighbors (kNN)": KNeighborsClassifier(),
            "Support Vector Machine (SVM)": SVC(random_state=42, probability=True), "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
            "Random Forest Classifier": RandomForestClassifier(random_state=42), "Naive Bayes (Gaussian)": GaussianNB(),
            "AdaBoost Classifier": AdaBoostClassifier(random_state=42), "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
            "XGBoost Classifier": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            "LightGBM Classifier": lgb.LGBMClassifier(random_state=42), "CatBoost Classifier": cb.CatBoostClassifier(verbose=0, random_state=42),
            "Linear Discriminant Analysis (LDA)": LinearDiscriminantAnalysis(), "Quadratic Discriminant Analysis (QDA)": QuadraticDiscriminantAnalysis(),
            "Perceptron": Perceptron(random_state=42), "Multi-Layer Perceptron (MLP)": MLPClassifier(random_state=42, max_iter=500),
            "Extra Trees Classifier": ExtraTreesClassifier(random_state=42),
        }
    },
    "Unsupervised Learning": {
        "Dimensionality Reduction": {
            "Principal Component Analysis (PCA)": PCA(n_components=2, random_state=42), "Kernel PCA": KernelPCA(n_components=2, kernel='rbf', random_state=42),
            "Independent Component Analysis (ICA)": FastICA(n_components=2, random_state=42, max_iter=1000), "t-SNE": TSNE(n_components=2, random_state=42),
            "Factor Analysis": FactorAnalysis(n_components=2, random_state=42), "Non-negative Matrix Factorization (NMF)": NMF(n_components=2, random_state=42, init='random'),
        }
    },
    "Semi-Supervised Learning": {"Classification": {"Label Propagation": LabelPropagation(), "Label Spreading": LabelSpreading()}}
}

ALGORITHM_EXPLANATIONS = {
    "Linear Regression": "Predicts a continuous value by fitting a straight line to the data. It's simple, fast, and great for understanding basic relationships.",
    "Logistic Regression": "A fundamental classification algorithm that predicts the probability of an outcome (e.g., yes/no). It works by drawing a line to separate different classes.",
    "Decision Tree": "Creates a tree-like model of decisions. It's easy to understand and visualize, as it mimics human decision-making with a series of if/else questions.",
    "Random Forest": "An 'ensemble' method that builds many decision trees and combines their outputs. This makes it much more accurate and robust than a single tree.",
    "Gradient Boosting / GBR / XGBoost / LightGBM / CatBoost": "A powerful family of 'ensemble' algorithms that build trees one after another, where each new tree corrects the errors of the previous one. They are known for extremely high performance.",
    "Support Vector Machine (SVM / SVR)": "Finds the best possible line or boundary to separate data points into classes or to fit a regression line. It's very effective in high-dimensional spaces.",
    "k-Nearest Neighbors (kNN)": "A simple algorithm that classifies a new data point based on the majority class of its 'k' closest neighbors. It's like judging a person by the company they keep.",
    "Naive Bayes": "A classification technique based on Bayes' Theorem. It's particularly useful for text classification problems like spam detection.",
    "Neural Network / MLP": "Inspired by the human brain, it consists of interconnected layers of 'neurons' that learn complex patterns from data. It's the foundation of deep learning.",
    "Principal Component Analysis (PCA)": "Reduces the number of features (dimensions) in a dataset while retaining as much information as possible. It finds new, uncorrelated 'principal components' that capture the most variance.",
    "Independent Component Analysis (ICA)": "Separates a multivariate signal into additive subcomponents, assuming that the source signals are non-Gaussian and statistically independent from each other.",
    "t-SNE": "A non-linear dimensionality reduction technique well-suited for embedding high-dimensional data for visualization in a low-dimensional space (e.g., 2D or 3D). It excels at preserving local structures.",
    "Factor Analysis": "A statistical method used to describe variability among observed, correlated variables in terms of a potentially lower number of unobserved variables called factors.",
    "Non-negative Matrix Factorization (NMF)": "A group of algorithms in multivariate analysis and linear algebra for signal processing, where a matrix V is factorized into (usually) two matrices W and H, with the property that all three matrices have no negative elements. This non-negativity makes the resulting matrices easier to inspect."
}

UNSUPERVISED_USE_CASES = {
    "Principal Component Analysis (PCA)": """
    - **Data Compression:** Reducing storage space by storing only the key components.
    - **Visualization:** Plotting complex, high-dimensional data in 2D or 3D to spot trends.
    - **Noise Reduction:** Removing less important components that might represent noise in the data.
    - **Improved Model Performance:** Used as a preprocessing step to improve the speed and accuracy of supervised models.
    """,
    "t-SNE": """
    - **Advanced Visualization:** Excellent for visualizing the structure of very high-dimensional data, often revealing clusters that PCA might miss.
    - **Scientific Research:** Used in fields like bioinformatics and cancer research to visualize relationships between cell populations.
    - **Machine Learning Debugging:** Helps to understand how data is clustered and why a model might be performing a certain way.
    """,
    "Independent Component Analysis (ICA)": """
    - **Signal Separation:** The classic example is separating individual voices from a recording of multiple people speaking at once (the "cocktail party problem").
    - **Medical Imaging:** Analyzing medical signals like EEG or fMRI data to isolate different sources of brain activity.
    - **Feature Extraction:** Identifying underlying independent factors in financial data.
    """
    # Add more for other unsupervised algorithms as needed
}

def get_general_explanation(model_name):
    """Finds a general explanation for a given model name, handling variations."""
    # Specific keywords for common ensembles
    if any(k in model_name for k in ["XGBoost", "LightGBM", "CatBoost", "Gradient Boosting", "AdaBoost"]):
        return ALGORITHM_EXPLANATIONS.get("Gradient Boosting / GBR / XGBoost / LightGBM / CatBoost", "A powerful ensemble boosting algorithm.")
    # Specific keywords for Tree-based
    if "Decision Tree" in model_name:
        return ALGORITHM_EXPLANATIONS.get("Decision Tree", "A tree-based algorithm.")
    if "Random Forest" in model_name:
        return ALGORITHM_EXPLANATIONS.get("Random Forest", "An ensemble of decision trees.")
    # General keyword matching for other models
    for key, value in ALGORITHM_EXPLANATIONS.items():
        if key in model_name:
            return value
    return "No specific explanation available for this model, but it is a powerful algorithm for this type of task."


# --- Helper Functions ---
def detect_problem_type(df, target_column):
    if target_column not in df.columns: return None
    target_series = df[target_column]
    if pd.api.types.is_float_dtype(target_series): return "Regression"
    if pd.api.types.is_object_dtype(target_series) or pd.api.types.is_categorical_dtype(target_series): return "Classification"
    if pd.api.types.is_integer_dtype(target_series):
        return "Regression" if target_series.nunique() > 25 else "Classification"
    return None

def build_preprocessor(df):
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numerical_cols),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_cols)
    ], remainder='passthrough')
    return preprocessor

@st.cache_data(show_spinner="Running all compatible models...")
def run_model_comparison(df, target_column, task):
    results = []
    compatible_models = MODELS["Supervised Learning"][task]
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    preprocessor = build_preprocessor(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for name, model in compatible_models.items():
        try:
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            score = r2_score(y_test, preds) if task == "Regression" else accuracy_score(y_test, preds)
            results.append({"Model": name, "Score": score})
        except Exception as e:
            results.append({"Model": name, "Score": f"Error: {e}"})

    metric_name = "R-squared" if task == "Regression" else "Accuracy"
    results_df = pd.DataFrame(results).sort_values(by="Score", ascending=False).reset_index(drop=True)
    results_df.rename(columns={"Score": metric_name}, inplace=True)
    return results_df

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="AutoML Explorer")
st.title("🚀 AutoML Explorer")
st.write("An intelligent tool to analyze data, find the best machine learning model, and understand the results.")

# Initialize session state for consistent data across tabs
if 'df' not in st.session_state: st.session_state.df = None
if 'learning_type' not in st.session_state: st.session_state.learning_type = None
if 'task_type' not in st.session_state: st.session_state.task_type = None
if 'model_name' not in st.session_state: st.session_state.model_name = None
if 'target_column' not in st.session_state: st.session_state.target_column = None
if 'unlabeled_value' not in st.session_state: st.session_state.unlabeled_value = "-1"
if 'run_analysis_clicked' not in st.session_state: st.session_state.run_analysis_clicked = False

# Results storage for the main run
if 'experiment_result' not in st.session_state: st.session_state.experiment_result = None
if 'transformed_df' not in st.session_state: st.session_state.transformed_df = None
if 'pca_variance' not in st.session_state: st.session_state.pca_variance = None
if 'best_supervised_model_name' not in st.session_state: st.session_state.best_supervised_model_name = None
if 'supervised_results_df' not in st.session_state: st.session_state.supervised_results_df = None


with st.sidebar:
    st.header("1. Upload Your Data")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df # Store df in session state
        st.header("2. Configure Your Model")
        
        st.session_state.learning_type = st.selectbox("Select Learning Type", list(MODELS.keys()), key='learning_type_sb')
        
        if st.session_state.learning_type == "Unsupervised Learning":
            st.session_state.task_type = st.selectbox("Select Task", list(MODELS[st.session_state.learning_type].keys()), key='unsupervised_task_sb')
            st.session_state.model_name = st.selectbox("Select Model", list(MODELS[st.session_state.learning_type][st.session_state.task_type].keys()), key='unsupervised_model_sb')
            st.session_state.target_column = None # No target for unsupervised
        else: # Supervised and Semi-Supervised
            st.session_state.target_column = st.selectbox("Select Target Column", [""] + df.columns.tolist(), key='target_column_sb')
            
            if st.session_state.target_column:
                detected_task = detect_problem_type(df, st.session_state.target_column)
                st.success(f"💡 Detected Task: **{detected_task}**")
                
                compatible_tasks = [t for t in MODELS[st.session_state.learning_type].keys() if t == detected_task]
                if compatible_tasks:
                    st.session_state.task_type = st.selectbox("Select Compatible Task", compatible_tasks, key='compatible_task_sb')
                    st.session_state.model_name = st.selectbox("Select Model", list(MODELS[st.session_state.learning_type][st.session_state.task_type].keys()), key='model_name_sb')
                else:
                    st.warning(f"No compatible tasks were found in '{st.session_state.learning_type}' for a '{detected_task}' problem.")
            
            if st.session_state.learning_type == "Semi-Supervised Learning" and st.session_state.target_column:
                st.session_state.unlabeled_value = st.text_input("Value for unlabeled data in target", "-1", key='unlabeled_value_ti')

        st.session_state.run_analysis_clicked = st.button("🚀 Run Analysis")

# --- Main Page Tabs ---
if st.session_state.df is not None:
    tab1, tab2, tab3 = st.tabs(["🧪 Experiment & Plots", "🏆 Best Model Finder", "📖 Analysis & Explanations"])

    if st.session_state.run_analysis_clicked:
        with st.spinner('Performing analysis...'):
            current_df = st.session_state.df.copy()
            preprocessor = build_preprocessor(current_df)
            
            # --- Perform the Experiment and store results ---
            if st.session_state.learning_type == "Unsupervised Learning":
                model_instance = MODELS[st.session_state.learning_type][st.session_state.task_type][st.session_state.model_name]
                pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model_instance)])
                transformed_data = pipeline.fit_transform(current_df)
                st.session_state.transformed_df = pd.DataFrame(transformed_data, columns=[f'Component {i+1}' for i in range(transformed_data.shape[1])])
                
                if isinstance(model_instance, PCA):
                    st.session_state.pca_variance = pipeline.named_steps['model'].explained_variance_ratio_
                else:
                    st.session_state.pca_variance = None # Reset if not PCA
                st.session_state.experiment_result = f"Successfully reduced data to {transformed_data.shape[1]} components."

            elif st.session_state.learning_type == "Supervised Learning":
                X = current_df.drop(st.session_state.target_column, axis=1)
                y = current_df[st.session_state.target_column]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model_instance = MODELS[st.session_state.learning_type][st.session_state.task_type][st.session_state.model_name]
                pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model_instance)])
                pipeline.fit(X_train, y_train)
                preds = pipeline.predict(X_test)
                
                if st.session_state.task_type == "Classification":
                    score = accuracy_score(y_test, preds)
                    metric_name = "Accuracy Score"
                else: # Regression
                    score = r2_score(y_test, preds)
                    metric_name = "R-squared (R²)"
                st.session_state.experiment_result = {"name": st.session_state.model_name, "score": score, "metric": metric_name}
                
                # Also run supervised comparison for Tab 2
                st.session_state.supervised_results_df = run_model_comparison(current_df, st.session_state.target_column, st.session_state.task_type)
                st.session_state.best_supervised_model_name = st.session_state.supervised_results_df.iloc[0]['Model']

            elif st.session_state.learning_type == "Semi-Supervised Learning":
                try:
                    val = int(st.session_state.unlabeled_value)
                except (ValueError, TypeError):
                    st.session_state.experiment_result = "Error: Unlabeled value must be an integer (e.g., -1)."
                else:
                    X = current_df.drop(st.session_state.target_column, axis=1)
                    y = current_df[st.session_state.target_column].copy()
                    unlabeled_mask = (y == val)
                    y[unlabeled_mask] = -1
                    X_processed = preprocessor.fit_transform(X)
                    model_instance = MODELS[st.session_state.learning_type][st.session_state.task_type][st.session_state.model_name]
                    model_instance.fit(X_processed, y)
                    current_df['predicted_labels'] = model_instance.transduction_
                    st.session_state.transformed_df = current_df[[st.session_state.target_column, 'predicted_labels']].head(10)
                    st.session_state.experiment_result = "Successfully propagated labels to unlabeled data."
            st.success("Analysis complete!")

    # --- Tab 1: Experiment & Plots ---
    with tab1:
        st.header("🧪 Experiment Results & Plots")
        st.subheader(f"Model: {st.session_state.model_name}")

        if st.session_state.learning_type == "Unsupervised Learning" and st.session_state.transformed_df is not None:
            st.write(st.session_state.experiment_result)
            st.write("Transformed Data (Head):", st.session_state.transformed_df.head())
            if st.session_state.transformed_df.shape[1] >= 2:
                fig = px.scatter(st.session_state.transformed_df, x='Component 1', y='Component 2', title=f"Dataset plotted by {st.session_state.model_name}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("At least 2 components are needed to plot a scatter graph.")
        elif st.session_state.learning_type == "Supervised Learning" and st.session_state.experiment_result:
            res = st.session_state.experiment_result
            st.metric(label=f"{res['name']} ({res['metric']})", value=f"{res['score']:.4f}")
        elif st.session_state.learning_type == "Semi-Supervised Learning" and st.session_state.transformed_df is not None:
            st.write(st.session_state.experiment_result)
            st.write("Data with Original and Predicted Labels (Head):", st.session_state.transformed_df)
        elif st.session_state.run_analysis_clicked and not st.session_state.experiment_result:
             st.warning("No results to display. Please ensure a valid model and target are selected and analysis runs successfully.")
        
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state.df.head())

    # --- Tab 2: Best Model Finder (Supervised only) ---
    with tab2:
        st.header("🏆 Best Model Finder")
        st.write("This tab automatically tests all compatible **Supervised Learning** models and ranks them by performance.")
        
        if st.session_state.learning_type != "Supervised Learning":
            st.info("Best Model Finder is only available for Supervised Learning tasks.")
        elif not st.session_state.run_analysis_clicked:
            st.info("Click 'Run Analysis' in the sidebar to populate this section.")
        elif st.session_state.supervised_results_df is not None:
            results = st.session_state.supervised_results_df
            best_model = results.iloc[0]
            metric_name = results.columns[1]
            st.success(f"🏆 The Best Model is **{best_model['Model']}** with a score of **{best_model[metric_name]:.4f}**.")
            st.subheader("Model Leaderboard")
            st.dataframe(results)
            st.subheader("Performance Comparison")
            chart_df = results.set_index("Model")
            chart_df = chart_df[pd.to_numeric(chart_df[metric_name], errors='coerce').notna()]
            st.bar_chart(chart_df)
        else:
            st.warning("No supervised learning results available to display yet.")


    # --- Tab 3: Analysis & Explanations ---
    with tab3:
        st.header("📖 Analysis & Explanations")
        if not st.session_state.run_analysis_clicked:
            st.info("Click 'Run Analysis' in the sidebar to populate this section.")
        else:
            st.subheader("Data Preprocessing Steps")
            st.markdown("""
            Before training any models, your data went through several automated preparation steps. This is crucial for ensuring accuracy and reliability.
            
            **1. Imputation (Handling Missing Values)**
            - **What:** The process filled in any empty cells in your dataset.
            - **Why:** Most machine learning algorithms cannot work with missing data.
            - **How:** For numerical columns, missing values were replaced with the **median** (the middle value). For categorical columns, they were replaced with the **mode** (the most frequent value). This is a robust strategy that isn't easily skewed by extreme outliers.

            **2. Scaling (Standardizing Numerical Features)**
            - **What:** All numerical columns were rescaled to have a mean of 0 and a standard deviation of 1.
            - **Why:** This prevents algorithms from being biased towards features with larger numerical ranges. For example, a salary column (e.g., 50000) would otherwise have more influence than an age column (e.g., 45) just because of its scale.
            
            **3. One-Hot Encoding (Handling Categorical Features)**
            - **What:** Text-based columns (like 'Color' or 'City') were converted into a numerical format that the algorithm can understand.
            - **Why:** Algorithms require all input to be numbers.
            - **How:** This process creates new binary (0 or 1) columns for each category. For a 'Color' column with 'Red' and 'Blue', it would create two new columns: `is_Red` and `is_Blue`. If a row's color was 'Red', the `is_Red` column would be 1 and `is_Blue` would be 0.
            """)

            if st.session_state.learning_type == "Unsupervised Learning":
                st.subheader(f"Explanation for {st.session_state.model_name}")
                st.write(get_general_explanation(st.session_state.model_name))
                
                st.subheader("Interpreting the Graph from Tab 1")
                if st.session_state.model_name == "Principal Component Analysis (PCA)" and st.session_state.pca_variance is not None:
                    variance = st.session_state.pca_variance
                    if len(variance) >= 2:
                        st.markdown(f"""
                        The scatter plot in Tab 1 visualizes the main patterns found in your dataset using **{st.session_state.model_name}**. We reduced your data's many features into just two essential components.
                        - **Component 1** (the horizontal axis) captures **{variance[0]:.2%}** of the total variance (information) in your data.
                        - **Component 2** (the vertical axis) captures an additional **{variance[1]:.2%}** of the variance.
                        
                        Together, these two components represent **{sum(variance):.2%}** of the original dataset's information. Data points that are close together on this graph are more similar to each other than points that are far apart.
                        """)
                    else:
                         st.markdown(f"""
                        The plot in Tab 1 visualizes your data reduced to a single component using **{st.session_state.model_name}**. This component captures **{variance[0]:.2%}** of the original dataset's information.
                        """)
                else:
                    st.markdown(f"""
                    The plot in Tab 1 visualizes the main patterns found in your dataset after being processed by the **{st.session_state.model_name}** algorithm. This method reduced your data's many features into just two components to make it plottable.
                    
                    Data points that are close together on this graph are more similar to each other than points that are far apart. You can look for clusters or groups in the plot to identify natural segments in your data.
                    """)
                
                st.subheader("Common Use Cases for this Algorithm")
                use_cases = UNSUPERVISED_USE_CASES.get(st.session_state.model_name, "No specific use cases listed for this algorithm.")
                st.markdown(use_cases)
            
            elif st.session_state.learning_type == "Supervised Learning" and st.session_state.best_supervised_model_name:
                st.subheader(f"Explanation for the Best Model: {st.session_state.best_supervised_model_name}")
                st.write(get_general_explanation(st.session_state.best_supervised_model_name))
            
            elif st.session_state.learning_type == "Semi-Supervised Learning":
                st.subheader(f"Explanation for {st.session_state.model_name}")
                st.write(get_general_explanation(st.session_state.model_name))
                st.markdown("""
                **Semi-Supervised Learning** algorithms like Label Propagation and Label Spreading leverage both labeled and unlabeled data for training. They typically work by:
                1. **Building a graph:** Connecting data points based on their similarity.
                2. **Propagating labels:** Known labels (from the labeled data) spread through the graph to nearby unlabeled points.
                3. **Iterative Refinement:** This process is repeated until the labels stabilize.
                This allows the model to learn from a larger dataset, even with limited initial labels, often leading to better performance than purely supervised methods when labeled data is scarce.
                """)
else:
    st.info("Upload a CSV file in the sidebar to get started.")