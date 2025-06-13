import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import time

# Set page configuration
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom styling
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #0066cc; text-align: center; margin-bottom: 30px;}
    .prediction-result {font-size: 1.8rem; font-weight: bold; text-align: center; padding: 15px; border-radius: 10px; margin: 10px 0;}
    .model-metrics {background-color: #f5f5f5; padding: 15px; border-radius: 10px; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

# Function to load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv('C:\\Users\\User\\Downloads\\medical_insurance (1).csv')
    return data

# Function to create and train models
@st.cache_resource
def train_model(model_name, X_train, y_train):
    if model_name == 'Linear Regression':
        model = LinearRegression()
    elif model_name == 'Ridge Regression':
        model = Ridge(alpha=1.0)
    elif model_name == 'Lasso Regression':
        model = Lasso(alpha=0.1)
    elif model_name == 'Decision Tree':
        model = DecisionTreeRegressor(random_state=42, max_depth=10)
    elif model_name == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == 'Gradient Boosting':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif model_name == 'XGBoost':
        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    elif model_name == 'SVR':
        model = SVR(kernel='rbf', C=1000, gamma=0.1)
    elif model_name == 'K-Nearest Neighbors':
        model = KNeighborsRegressor(n_neighbors=5)
    
    # Create preprocessing pipeline
    numeric_features = ['age', 'bmi', 'children']
    categorical_features = ['sex', 'smoker', 'region']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    return pipeline

# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return {
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'MAE': mae
    }

# Function to display feature importance
def plot_feature_importance(model, feature_names):
    if hasattr(model['model'], 'feature_importances_'):
        preprocessor = model['preprocessor']
        ohe = preprocessor.transformers_[1][1]
        cat_features = preprocessor.transformers_[1][2]
        
        # Get feature names after one-hot encoding
        ohe_feature_names = ohe.get_feature_names_out(cat_features)
        all_feature_names = list(preprocessor.transformers_[0][2]) + list(ohe_feature_names)
        
        # Get feature importances
        importances = model['model'].feature_importances_
        
        # Create DataFrame for plotting
        feature_imp = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': importances
        })
        feature_imp = feature_imp.sort_values('Importance', ascending=False)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_imp, ax=ax)
        ax.set_title('Feature Importance')
        plt.tight_layout()
        return fig
    return None

# Main application
def main():
    # Load data
    data = load_data()
    
    # Title
    st.markdown("<h1 class='main-header'>Medical Insurance Cost Predictor</h1>", unsafe_allow_html=True)

    # Sidebar for model selection
    st.sidebar.title("Model Selection")
    
    available_models = [
        'Linear Regression',
        'Ridge Regression',
        'Lasso Regression',
        'Decision Tree',
        'Random Forest',
        'Gradient Boosting',
        'XGBoost',
        'SVR',
        'K-Nearest Neighbors'
    ]
    
    selected_models = st.sidebar.multiselect(
        "Select models to compare:",
        available_models,
        default=['Random Forest', 'Linear Regression']
    )
    
    # Split data
    X = data.drop('charges', axis=1)
    y = data['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # User input section
    st.write("### Enter Your Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        sex = st.selectbox("Sex", options=["male", "female"])
    
    with col2:
        bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=27.5, step=0.1)
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=1)
    
    with col3:
        smoker = st.selectbox("Smoker", options=["no", "yes"])
        region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])
    
    # Create input data for prediction
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    
    # Model training and prediction
    if st.button("Predict Insurance Cost"):
        if not selected_models:
            st.error("Please select at least one model!")
        else:
            # Create tabs for model results
            tabs = st.tabs(selected_models)
            
            # Container for all predictions
            all_predictions = {}
            
            # Train models and make predictions
            for i, model_name in enumerate(selected_models):
                with tabs[i]:
                    with st.spinner(f"Training {model_name}..."):
                        # Train model
                        start_time = time.time()
                        model = train_model(model_name, X_train, y_train)
                        training_time = time.time() - start_time
                        
                        # Make prediction
                        prediction = model.predict(input_data)[0]
                        all_predictions[model_name] = prediction
                        
                        # Display prediction
                        st.markdown(f"<div class='prediction-result' style='background-color: #0066cc;'>Predicted Insurance Cost: ${prediction:.2f}</div>", unsafe_allow_html=True)
                        
                        # Evaluate model
                        metrics = evaluate_model(model, X_test, y_test)
                        
                        # Display metrics
                        st.write("### Model Performance Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("MSE", f"{metrics['MSE']:.2f}")
                        col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
                        col3.metric("R²", f"{metrics['R²']:.4f}")
                        col4.metric("MAE", f"{metrics['MAE']:.2f}")
                        
                        st.write(f"Training Time: {training_time:.2f} seconds")
                        
                        # Plot feature importance (if available)
                        if model_name in ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost']:
                            st.write("### Feature Importance")
                            fig = plot_feature_importance(model, X.columns)
                            if fig:
                                st.pyplot(fig)
            
            # Compare models
            if len(selected_models) > 1:
                st.write("## Model Comparison")
                
                # Create comparison dataframe
                comparison_df = pd.DataFrame({
                    'Model': list(all_predictions.keys()),
                    'Predicted Cost ($)': list(all_predictions.values())
                })
                comparison_df = comparison_df.sort_values('Predicted Cost ($)', ascending=True)
                
                # Plot comparison
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = sns.barplot(x='Model', y='Predicted Cost ($)', data=comparison_df, ax=ax)
                
                # Add data labels
                for i, bar in enumerate(bars.patches):
                    bars.text(
                        bar.get_x() + bar.get_width()/2.,
                        bar.get_height() + 500,
                        f"${comparison_df['Predicted Cost ($)'].iloc[i]:.2f}",
                        ha='center',
                        va='bottom'
                    )
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display table
                st.write("### Detailed Comparison")
                st.table(comparison_df)
    
    # Dataset exploration section
    if st.sidebar.checkbox("Show Dataset Information"):
        st.sidebar.write("### Dataset Overview")
        st.sidebar.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
        
        # Show sample data
        if st.sidebar.checkbox("Show Sample Data"):
            st.sidebar.dataframe(data.head())
            
        # Show feature distributions
        if st.sidebar.checkbox("Show Feature Distributions"):
            feature = st.sidebar.selectbox("Select Feature", options=data.columns)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(data[feature], kde=True, ax=ax)
            st.sidebar.pyplot(fig)
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This application predicts medical insurance costs using various machine learning models. "
        "You can select multiple models to compare their predictions and performance metrics."
    )

if __name__ == "__main__":
    main()