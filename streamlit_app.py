import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import altair as alt
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Page title
st.set_page_config(page_title='PCA and Ensemble Learning', page_icon='🤖')
st.title('🤖 PCA and Ensemble Learning')
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
        - Voting: Combines predictions from multiple models (classifiers) using majority voting.
        - Bagging: Reduces variance by training multiple models on different subsets of data.
        - Boosting: Combines models sequentially, each correcting errors of the previous ones.
        - Stacking: Trains multiple models and combines their predictions using a meta-model.
        ''')

# Sidebar for uploading data
with st.sidebar:
    st.header('Upload Data')
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

analysis_type = st.selectbox('Select Analysis Type', ['PCA', 'Ensemble Learning'])


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data preview:")
    st.write(df.head())

    # User selects analysis type

    if analysis_type == 'PCA':
        st.header('Principal Component Analysis (PCA)')
        st.info('Select numerical features for PCA. Non-numerical data types are not supported.')
        features = st.multiselect('Select features for PCA', df.columns.tolist())

        if features:
            X = df[features]
            st.write("Features preview:")
            st.write(X.head())

            # Handle missing values
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)

            st.header('PCA Analysis')
            n_components = st.slider('Number of PCA components', 2, min(X.shape[1], 10), 2, 1)
            st.warning('PCA requires components between 2 and the smaller value between the number of features in the dataset and 10.')

            # Apply PCA
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(X)
            explained_variance = pca.explained_variance_ratio_
            st.write(f'Explained variance by {n_components} components: {explained_variance.sum():.2f}')
            pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
            st.write(pca_df.head())

            # Visualize the first two principal components
            if n_components > 1:
                st.header('PCA Results Visualization')
                pca_chart = alt.Chart(pca_df).mark_circle(size=60).encode(
                    x='PC1',
                    y='PC2',
                    tooltip=list(pca_df.columns)
                ).interactive()
                st.altair_chart(pca_chart, use_container_width=True)

            # Scree plot
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

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            if method == 'Voting':
                st.subheader('Voting Classifier')
                base_models = [
                    ('Logistic Regression', LogisticRegression()),
                    ('Decision Tree', DecisionTreeClassifier()),
                    ('SVM', SVC(probability=True)),
                    ('KNN', KNeighborsClassifier())
                ]
                voting_clf = VotingClassifier(estimators=base_models, voting='soft')
                voting_clf.fit(X_train, y_train)
                voting_pred = voting_clf.predict(X_test)
                voting_accuracy = accuracy_score(y_test, voting_pred)
                st.write(f'Voting Classifier Accuracy: {voting_accuracy:.2f}')

            elif method == 'Bagging':
                st.subheader('Bagging Classifier')
                bagging_clf = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42)
                bagging_clf.fit(X_train, y_train)
                bagging_pred = bagging_clf.predict(X_test)
                bagging_accuracy = accuracy_score(y_test, bagging_pred)
                st.write(f'Bagging Classifier Accuracy: {bagging_accuracy:.2f}')


            elif method == 'Boosting':
                st.subheader('Boosting Classifier')
                boosting_clf = AdaBoostClassifier(n_estimators=50, random_state=42)
                boosting_clf.fit(X_train, y_train)
                boosting_pred = boosting_clf.predict(X_test)
                boosting_accuracy = accuracy_score(y_test, boosting_pred)
                st.write(f'Boosting Classifier Accuracy: {boosting_accuracy:.2f}')

            elif method == 'Stacking':
                st.subheader('Stacking Classifier')
                base_models = [
                    ('Logistic Regression', LogisticRegression()),
                    ('Decision Tree', DecisionTreeClassifier()),
                    ('SVM', SVC(probability=True)),
                    ('KNN', KNeighborsClassifier())
                ]
                stacking_clf = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())
                stacking_clf.fit(X_train, y_train)
                stacking_pred = stacking_clf.predict(X_test)
                stacking_accuracy = accuracy_score(y_test, stacking_pred)
                st.write(f'Stacking Classifier Accuracy: {stacking_accuracy:.2f}')

else:
    st.warning('👈 Upload a CSV file to get started!')
