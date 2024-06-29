import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import altair as alt
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier, StackingClassifier, VotingRegressor, BaggingRegressor, AdaBoostRegressor, StackingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Page title
st.set_page_config(page_title='PCA and Ensemble Learning', page_icon='🤖')
st.title('🤖 PCA and Ensemble Learning')

# Expanders for PCA and Ensemble Learning explanations
with st.expander('What is Principal Component Analysis (PCA)?'):
    st.text('''
    Principal Component Analysis (PCA) is a statistical technique used to reduce
    the dimensionality of a dataset while retaining most of the variability in
    the data. It transforms the original variables into a new set of variables 
    called principal components, which are orthogonal (uncorrelated) and ordered 
    by the amount of variance they explain in the data.
    ''')

with st.expander('What is Ensemble Learning?'):
    st.text('''
    Ensemble learning combines multiple models to improve performance. Common methods include:
    - Voting: Combines predictions from multiple models (classifiers or regressors) using majority voting or averaging.
    - Bagging: Reduces variance by training multiple models on different subsets of data.
    - Boosting: Combines models sequentially, each correcting errors of the previous ones.
    - Stacking: Trains multiple models and combines their predictions using a meta-model.
    ''')

# Sidebar for uploading data and selecting analysis type
with st.sidebar:
    st.header('Upload Data')
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    analysis_type = st.selectbox('Select Analysis Type', ['PCA', 'Ensemble Learning'])
    if analysis_type == 'Ensemble Learning':
        task_type = st.selectbox('Select Task Type', ['Classification', 'Regression'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data preview:")
    st.write(df.head())

    if analysis_type == 'PCA':
        st.header('Principal Component Analysis (PCA)')
        st.info('Select numerical features for PCA. Non-numerical data types are not supported.')
        features = st.multiselect('Select features for PCA', df.columns.tolist())

        if features:
            X = df[features]
            st.write("Features preview:")
            st.write(X.head())

            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)

            st.header('PCA Analysis')
            n_components = st.slider('Number of PCA components', 2, min(X.shape[1], 10), 2, 1)
            st.warning('PCA requires components between 2 and the smaller value between the number of features in the dataset and 10.')

            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(X)
            explained_variance = pca.explained_variance_ratio_
            st.write(f'Explained variance by {n_components} components: {explained_variance.sum():.2f}')
            pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
            st.write(pca_df.head())

            if n_components > 1:
                st.header('PCA Results Visualization')
                pca_chart = alt.Chart(pca_df).mark_circle(size=60).encode(
                    x='PC1',
                    y='PC2',
                    tooltip=list(pca_df.columns)
                ).interactive()
                st.altair_chart(pca_chart, use_container_width=True)

            st.header('Scree Plot')
            fig, ax = plt.subplots(figsize=(10, 6))
            components = np.arange(1, len(explained_variance) + 1)
            ax.bar(components, explained_variance * 100, alpha=0.6, align='center', label='Individual Explained Variance')
            ax.plot(components, np.cumsum(explained_variance) * 100, marker='o', color='black', label='Cumulative Explained Variance')
            for i, v in enumerate(explained_variance * 100):
                ax.text(i + 1, v + 1, f"{v:.1f}%", ha='center', va='bottom')
            ax.set_xlabel('Dimensions')
            ax.set_ylabel('Percentage of Explained Variance')
            ax.set_title('Scree Plot')
            ax.legend(loc='best')
            ax.grid(True)
            st.pyplot(fig)

            st.header('Conclusion')
            st.write(f"The selected principal components explain {explained_variance.sum():.2%} of the variance in the data.")
            if explained_variance.sum() < 0.80:
                st.write("Consider adding more components or preprocessing your data further for better results.")
            else:
                st.write("The PCA results indicate a significant reduction in dimensionality with minimal loss of information.")

    elif analysis_type == 'Ensemble Learning':
        st.header('Ensemble Learning Methods')
        method = st.selectbox('Select Ensemble Method', ['Voting', 'Bagging', 'Boosting', 'Stacking'])

        with st.sidebar:
            st.info("Select numerical features for the model.")
            features = st.multiselect('Select features for Ensemble Learning', df.columns.tolist())
            target = st.selectbox('Select target variable', [col for col in df.columns if col not in features])

        if features and target:
            X = df[features]
            y = df[target]
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            if task_type == 'Classification':
                if method == 'Voting':
                    st.subheader('Voting Classifier')
                    st.text('''
                    Voting Classifier combines predictions from multiple models using majority voting (hard voting) or average of probabilities (soft voting).
                    - Hard voting: Majority class is predicted.
                    - Soft voting: Weighted average of probabilities is used.
                    ''')
                    voting_type = st.radio("Select voting type", ('hard', 'soft'))
                    base_models = [
                        ('Logistic Regression', LogisticRegression()),
                        ('Decision Tree', DecisionTreeClassifier()),
                        ('SVM', SVC(probability=True)),
                        ('KNN', KNeighborsClassifier())
                    ]
                    voting_clf = VotingClassifier(estimators=base_models, voting=voting_type)
                    voting_clf.fit(X_train, y_train)
                    voting_pred = voting_clf.predict(X_test)
                    voting_accuracy = accuracy_score(y_test, voting_pred)
                    st.write(f'Voting Classifier Accuracy: {voting_accuracy:.2f}')

                    st.header('Conclusion')
                    st.write('The Voting Classifier combines predictions from Logistic Regression, Decision Tree, SVM, and KNN models. It is beneficial when different models capture different aspects of the data.')

                elif method == 'Bagging':
                    st.subheader('Bagging Classifier')
                    st.text('''
                    Bagging Classifier reduces variance by training multiple models on different subsets of data.
                    - n_estimators: Number of models to train.
                    - max_samples: Proportion of the dataset used for each model.
                    ''')
                    n_estimators = st.slider('Number of estimators', 1, 100, 10)
                    max_samples = st.slider('Max samples', 0.1, 1.0, 0.5)
                    bagging_clf = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=n_estimators, max_samples=max_samples, random_state=42)
                    bagging_clf.fit(X_train, y_train)
                    bagging_pred = bagging_clf.predict(X_test)
                    bagging_accuracy = accuracy_score(y_test, bagging_pred)
                    st.write(f'Bagging Classifier Accuracy: {bagging_accuracy:.2f}')

                    st.header('Conclusion')
                    st.write('The Bagging Classifier uses multiple Decision Trees trained on different subsets of data. It helps reduce variance and overfitting.')

                elif method == 'Boosting':
                    st.subheader('Boosting Classifier')
                    st.text('''
                    Boosting Classifier improves accuracy by combining multiple weak learners sequentially.
                    - n_estimators: Number of boosting stages to run.
                    - learning_rate: Step size for updating weights.
                    ''')
                    n_estimators = st.slider('Number of estimators', 1, 100, 50)
                    learning_rate = st.slider('Learning rate', 0.01, 1.0, 0.1)
                    boosting_clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
                    boosting_clf.fit(X_train, y_train)
                    boosting_pred = boosting_clf.predict(X_test)
                    boosting_accuracy = accuracy_score(y_test, boosting_pred)
                    st.write(f'Boosting Classifier Accuracy: {boosting_accuracy:.2f}')

                    st.header('Conclusion')
                    st.write('The Boosting Classifier uses AdaBoost to combine multiple weak learners. It is effective for improving accuracy by focusing on difficult cases.')

                elif method == 'Stacking':
                    st.subheader('Stacking Classifier')
                    st.text('''
                    Stacking Classifier combines predictions from multiple models using a meta-model. It leverages the strengths of different models.
                    - Base models: Individual models used for predictions.
                    - Final estimator: Model used to combine predictions of base models.
                    ''')
                    base_models = [
                        ('Logistic Regression', LogisticRegression()),
                        ('Decision Tree', DecisionTreeClassifier()),
                        ('SVM', SVC(probability=True)),
                        ('KNN', KNeighborsClassifier())
                    ]
                    final_estimator = st.selectbox('Select final estimator', ['Logistic Regression', 'Decision Tree'])
                    if final_estimator == 'Logistic Regression':
                        final_model = LogisticRegression()
                    else:
                        final_model = DecisionTreeClassifier()
                    
                    stacking_clf = StackingClassifier(estimators=base_models, final_estimator=final_model)
                    stacking_clf.fit(X_train, y_train)
                    stacking_pred = stacking_clf.predict(X_test)
                    stacking_accuracy = accuracy_score(y_test, stacking_pred)
                    st.write(f'Stacking Classifier Accuracy: {stacking_accuracy:.2f}')

                    st.header('Conclusion')
                    st.write('The Stacking Classifier combines the predictions of Logistic Regression, Decision Tree, SVM, and KNN models using a final model. It is suitable for complex problems where different models capture different patterns.')

            elif task_type == 'Regression':
                if method == 'Voting':
                    st.subheader('Voting Regressor')
                    st.text('''
                    Voting Regressor combines predictions from multiple models using average voting.
                    ''')
                    base_models = [
                        ('Linear Regression', LinearRegression()),
                        ('Decision Tree', DecisionTreeRegressor()),
                        ('SVR', SVR()),
                        ('KNN', KNeighborsRegressor())
                    ]
                    voting_reg = VotingRegressor(estimators=base_models)
                    voting_reg.fit(X_train, y_train)
                    voting_pred = voting_reg.predict(X_test)
                    voting_mse = mean_squared_error(y_test, voting_pred)
                    st.write(f'Voting Regressor MSE: {voting_mse:.2f}')

                    st.header('Conclusion')
                    st.write('The Voting Regressor combines predictions from Linear Regression, Decision Tree, SVR, and KNN models. It helps improve stability and accuracy.')

                elif method == 'Bagging':
                    st.subheader('Bagging Regressor')
                    st.text('''
                    Bagging Regressor reduces variance by training multiple models on different subsets of data.
                    - n_estimators: Number of models to train.
                    - max_samples: Proportion of the dataset used for each model.
                    ''')
                    n_estimators = st.slider('Number of estimators', 1, 100, 10)
                    max_samples = st.slider('Max samples', 0.1, 1.0, 0.5)
                    bagging_reg = BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=n_estimators, max_samples=max_samples, random_state=42)
                    bagging_reg.fit(X_train, y_train)
                    bagging_pred = bagging_reg.predict(X_test)
                    bagging_mse = mean_squared_error(y_test, bagging_pred)
                    st.write(f'Bagging Regressor MSE: {bagging_mse:.2f}')

                    st.header('Conclusion')
                    st.write('The Bagging Regressor uses multiple Decision Trees trained on different subsets of data. It helps reduce variance and overfitting.')

                elif method == 'Boosting':
                    st.subheader('Boosting Regressor')
                    st.text('''
                    Boosting Regressor improves accuracy by combining multiple weak learners sequentially.
                    - n_estimators: Number of boosting stages to run.
                    - learning_rate: Step size for updating weights.
                    ''')
                    n_estimators = st.slider('Number of estimators', 1, 100, 50)
                    learning_rate = st.slider('Learning rate', 0.01, 1.0, 0.1)
                    boosting_reg = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
                    boosting_reg.fit(X_train, y_train)
                    boosting_pred = boosting_reg.predict(X_test)
                    boosting_mse = mean_squared_error(y_test, boosting_pred)
                    st.write(f'Boosting Regressor MSE: {boosting_mse:.2f}')

                    st.header('Conclusion')
                    st.write('The Boosting Regressor uses AdaBoost to combine multiple weak learners. It is effective for improving accuracy by focusing on difficult cases.')

                elif method == 'Stacking':
                    st.subheader('Stacking Regressor')
                    st.text('''
                    Stacking Regressor combines predictions from multiple models using a meta-model. It leverages the strengths of different models.
                    - Base models: Individual models used for predictions.
                    - Final estimator: Model used to combine predictions of base models.
                    ''')
                    base_models = [
                        ('Linear Regression', LinearRegression()),
                        ('Decision Tree', DecisionTreeRegressor()),
                        ('SVR', SVR()),
                        ('KNN', KNeighborsRegressor())
                    ]
                    final_estimator = st.selectbox('Select final estimator', ['Linear Regression', 'Decision Tree'])
                    if final_estimator == 'Linear Regression':
                        final_model = LinearRegression()
                    else:
                        final_model = DecisionTreeRegressor()
                    
                    stacking_reg = StackingRegressor(estimators=base_models, final_estimator=final_model)
                    stacking_reg.fit(X_train, y_train)
                    stacking_pred = stacking_reg.predict(X_test)
                    stacking_mse = mean_squared_error(y_test, stacking_pred)
                    st.write(f'Stacking Regressor MSE: {stacking_mse:.2f}')

                    st.header('Conclusion')
                    st.write('The Stacking Regressor combines the predictions of Linear Regression, Decision Tree, SVR, and KNN models using a final model. It is suitable for complex problems where different models capture different patterns.')
        else:
            st.warning('👈 Select features and target')
else:
    st.warning('👈 Upload a CSV file to get started!')
