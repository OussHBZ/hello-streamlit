import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import altair as alt

# Page title
st.set_page_config(page_title='PCA and Ensemble Analysis', page_icon='🤖')
st.title('🤖 PCA and Ensemble Models Analysis')

# Sidebar for uploading data
with st.sidebar:
    st.header('1. Upload Data')
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data preview:")
        st.write(df.head())
    else:
        st.warning('Please upload a CSV file to proceed.')
 
# Continue only if data is uploaded
if uploaded_file is not None:
    with st.sidebar:
        st.header('2. Choose Features and Target')
        all_columns = df.columns.tolist()
        features = st.multiselect('Select features for X', all_columns)
        target = st.selectbox('Select target variable for y', all_columns)

        if not features or not target:
            st.warning('Please select the features and target variable to proceed.')

        else:
            st.header('3. Choose Analysis Method')
            analysis_method = st.selectbox('Select analysis method', ['PCA', 'Ensemble Models'])

            if analysis_method == 'Ensemble Models':
                model_choice = st.selectbox('Choose Model', ['Random Forest', 'Gradient Boosting'])
                if model_choice == 'Random Forest':
                    parameter_n_estimators = st.slider('Number of estimators (n_estimators)', 10, 1000, 100, 10)
                    parameter_max_features = st.selectbox('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
                    parameter_min_samples_split = st.slider('Minimum samples to split (min_samples_split)', 2, 10, 2, 1)
                    parameter_min_samples_leaf = st.slider('Minimum samples at leaf (min_samples_leaf)', 1, 10, 1, 1)
                else:
                    parameter_n_estimators = st.slider('Number of estimators (n_estimators)', 10, 1000, 100, 10)
                    parameter_learning_rate = st.slider('Learning Rate (learning_rate)', 0.01, 1.0, 0.1, 0.01)
                    parameter_min_samples_split = st.slider('Minimum samples to split (min_samples_split)', 2, 10, 2, 1)
                    parameter_min_samples_leaf = st.slider('Minimum samples at leaf (min_samples_leaf)', 1, 10, 1, 1)

    if features and target:
        # Preprocess the data
        X = df[features]
        y = df[target]
        
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)

        if analysis_method == 'PCA':
            st.header('PCA Analysis')
            n_components = st.slider('Number of PCA components', 2, min(X.shape[1], 10), 2, 1)

            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(X)
            explained_variance = pca.explained_variance_ratio_

            st.write(f'Explained variance by {n_components} components: {explained_variance.sum():.2f}')
            pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
            st.write(pca_df.head())

            st.header('PCA Results Visualization')
            pca_chart = alt.Chart(pca_df).mark_circle(size=60).encode(
                x='PC1',
                y='PC2',
                tooltip=list(pca_df.columns)
            ).interactive()

            st.altair_chart(pca_chart, use_container_width=True)

        elif analysis_method == 'Ensemble Models':
            st.header('Ensemble Model Analysis')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if model_choice == 'Random Forest':
                model = RandomForestRegressor(
                    n_estimators=parameter_n_estimators,
                    max_features=parameter_max_features,
                    min_samples_split=parameter_min_samples_split,
                    min_samples_leaf=parameter_min_samples_leaf,
                    random_state=42
                )
            else:
                model = GradientBoostingRegressor(
                    n_estimators=parameter_n_estimators,
                    learning_rate=parameter_learning_rate,
                    min_samples_split=parameter_min_samples_split,
                    min_samples_leaf=parameter_min_samples_leaf,
                    random_state=42
                )

            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_mse = mean_squared_error(y_train, y_train_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            st.write(f'Training MSE: {train_mse:.2f}')
            st.write(f'Training R2: {train_r2:.2f}')
            st.write(f'Test MSE: {test_mse:.2f}')
            st.write(f'Test R2: {test_r2:.2f}')

            st.header('Feature Importance')
            feature_importance = model.feature_importances_
            features_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
            feature_chart = alt.Chart(features_df).mark_bar().encode(
                x='Importance',
                y=alt.Y('Feature', sort='-x')
            ).properties(height=400)

            st.altair_chart(feature_chart, use_container_width=True)

            st.header('Prediction Results')
            results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})
            results_chart = alt.Chart(results_df).mark_circle(size=60).encode(
                x='Actual',
                y='Predicted',
                tooltip=['Actual', 'Predicted']
            ).interactive()

            st.altair_chart(results_chart, use_container_width=True)
else:
    st.warning('👈 Upload a CSV file to get started!')
