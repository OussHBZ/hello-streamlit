import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import altair as alt
import matplotlib.pyplot as plt

# Page title
st.set_page_config(page_title='PCA Analysis', page_icon='🤖')
st.title('🤖 PCA Analysis')

  # PCA definition in an expander
with st.expander('What is Principal Component Analysis (PCA)?'):
        st.text('''
        Principal Component Analysis (PCA) is a statistical 
        technique used to reduce the dimensionality 
        of a dataset while retaining most of the variability 
        in the data. It transforms the original variables 
        into a new set of variables called principal 
        components, which are orthogonal (uncorrelated) 
        and ordered by the amount of variance they explain 
        in the data.
        ''')

# Sidebar for uploading data
with st.sidebar:
    st.header('Upload Data')
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Continue only if data is uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # Data preview
    st.write("Data preview:")
    st.write(df.head())

# Continue only if data is uploaded
if uploaded_file is not None:
    with st.sidebar:
        st.header('Choose Features for PCA')
        all_columns = df.columns.tolist()
        features = st.multiselect('Select features for PCA', all_columns)

        if not features:
            st.warning('Please select the features to proceed.')

    if features:
        # Preprocess the data
        X = df[features]
        
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        with st.sidebar:
            st.warning('The average is used to replace the missing values.')

        st.header('PCA Analysis')
        n_components = st.slider('Number of PCA components', 2, min(X.shape[1], 10), 2, 1)

        # Apply PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(X)
        explained_variance = pca.explained_variance_ratio_

        st.write(f'Explained variance by {n_components} components: {explained_variance.sum():.2f}')
        pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
        st.write(pca_df.head())

        st.header('PCA Results Visualization')
        if n_components > 1:
            pca_chart = alt.Chart(pca_df).mark_circle(size=60).encode(
                x='PC1',
                y='PC2',
                tooltip=list(pca_df.columns)
            ).interactive()

            st.altair_chart(pca_chart, use_container_width=True)
        
        # Scree plot
        st.header('Scree Plot')
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained')
        plt.title('Scree Plot')
        st.pyplot(plt)

        st.header('Conclusion')
        st.write(f"The selected principal components explain {explained_variance.sum():.2%} of the variance in the data.")
        
        if explained_variance.sum() < 0.80:
            st.write("Consider adding more components or preprocessing your data further for better results.")
        else:
            st.write("The PCA results indicate a significant reduction in dimensionality with minimal loss of information.")
else:
    st.warning('👈 Upload a CSV file to get started!')
