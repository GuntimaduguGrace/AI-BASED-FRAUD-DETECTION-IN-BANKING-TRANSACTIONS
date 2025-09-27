import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error
from streamlit_lottie import st_lottie
import json

# Set the style and layout for Streamlit app
st.set_page_config(layout="wide")
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (8, 6)

# Function to load and preprocess data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.drop('isFlaggedFraud', axis=1, inplace=True)
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        if df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='unsigned')
    df['type'] = df['type'].astype('category')
    return df

# Function to load Lottie animation
def load_lottie(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

# Sidebar configuration
st.sidebar.title("Navigation")
menu_options = ["Home", "Dataset Info", "Visualization", "Model Comparison"]
choice = st.sidebar.radio("Go to", menu_options)

# Load Lottie animation
lottie_path = "Animation - 1722944988921.json"
lottie_animation = load_lottie(lottie_path)

# Sidebar file uploader
st.sidebar.title("Upload CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = load_data(uploaded_file)

    if choice == "Home":
        st.title("Online Payment Fraud Detection Analysis")
        st_lottie(lottie_animation, height=300, key="animation")
        st.write("Welcome to the Online Payment Fraud Detection Analysis app. Use the sidebar to navigate through the sections.")

    elif choice == "Dataset Info":
        st.header("Dataset Info")
        st.write("Shape of the dataset:", df.shape)
        st.write(df.info())
        st.write("Sample Data")
        st.write(df.sample(5))

    elif choice == "Visualization":
        st.header("Data Visualization")

        st.subheader("Univariate Data Visualization")
        st.subheader("Transaction Type Distribution")
        fig, ax = plt.subplots()
        ax = sns.countplot(x='type', data=df, palette='PuBu')
        for container in ax.containers:
            ax.bar_label(container)
        plt.title('Count plot of transaction type')
        plt.ylabel('Number of transactions')
        st.pyplot(fig)

        st.subheader("Transaction Amount Distribution")
        fig, ax = plt.subplots()
        sns.kdeplot(df['amount'], linewidth=4)
        plt.title('Distribution of transaction amount')
        st.pyplot(fig)

        # Additional univariate visualizations...

        st.subheader("Bivariate Data Visualization")
        st.subheader("Transaction Type and Fraud Distribution")
        fig, ax = plt.subplots(1, 2, figsize=(20, 5))

        sns.countplot(x='type', data=df, hue='isFraud', palette='PuBu', ax=ax[0])
        for container in ax[0].containers:
            ax[0].bar_label(container)
        ax[0].set_title('Count plot of transaction type')
        ax[0].legend(loc='best')
        ax[0].set_ylabel('Number of transactions')

        df2 = df.groupby(['type', 'isFraud']).size().unstack()
        df2.apply(lambda x: round(x/sum(x)*100, 2), axis=1).plot(kind='barh', stacked=True, color=['lightsteelblue', 'steelblue'], ax=ax[1])
        for container in ax[1].containers:
            ax[1].bar_label(container, label_type='center')
        ax[1].set_title('Percentage of fraud by transaction type')
        ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax[1].set_ylabel('Transaction Type')
        ax[1].grid(axis='y')

        st.pyplot(fig)

        # Additional bivariate visualizations...

    elif choice == "Model Comparison":
        st.header("Model Comparison")

        # Data Preprocessing
        st.subheader("Data Preprocessing")

        # Splitting data into features and target
        X = df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
        y = df['isFraud']

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Add a button to train and evaluate models
        if st.button('Train and Evaluate Models'):
            # Reduce dataset size for faster computation
            sample_size = min(len(X_train), 10000)  # Limit to 10,000 samples
            X_train_sample, _, y_train_sample, _ = train_test_split(X_train, y_train, train_size=sample_size, random_state=42)

            # Train models and collect metrics
            results = {
                "Logistic Regression": {},
                "Random Forest": {},
                "SVM": {}
            }

            # Define models
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10),  # Reduced complexity
                "SVM": SVC(C=1.0, kernel='linear', max_iter=1000)  # Linear kernel and limited iterations
            }

            for model_name, model in models.items():
                st.subheader(f"{model_name} Model")
                model.fit(X_train_sample, y_train_sample)
                y_pred = model.predict(X_test)

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                mae = mean_absolute_error(y_test, y_pred)

                # Store results
                results[model_name]['Accuracy'] = accuracy
                results[model_name]['RMSE'] = rmse
                results[model_name]['MAE'] = mae

                st.write(f"*Accuracy:* {accuracy:.4f}")
                st.write(f"*RMSE:* {rmse:.4f}")
                st.write(f"*MAE:* {mae:.4f}")

                # Confusion matrix
                st.subheader(f"{model_name} Confusion Matrix")
                fig, ax = plt.subplots()
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt="d", cmap='coolwarm', linewidths=2, ax=ax)
                plt.title(f'{model_name} Confusion Matrix')
                st.pyplot(fig)

            # Comparison Plots
            st.subheader("Comparison of Model Performance")

            # Create DataFrame for comparison plots
            comparison_df = pd.DataFrame(results).T
            comparison_df = comparison_df[['Accuracy', 'RMSE', 'MAE']]

            # Accuracy Bar Plot
            st.subheader("Model Accuracy Comparison")
            fig, ax = plt.subplots()
            sns.barplot(x=comparison_df.index, y='Accuracy', data=comparison_df, palette='viridis', ax=ax)
            ax.set_title('Model Accuracy Comparison')
            ax.set_xlabel('Model')
            ax.set_ylabel('Accuracy')
            ax.set_ylim(0, 1)  # Ensure y-axis starts from 0
            st.pyplot(fig)

            # RMSE Bar Plot
            st.subheader("Model RMSE Comparison")
            fig, ax = plt.subplots()
            sns.barplot(x=comparison_df.index, y='RMSE', data=comparison_df, palette='viridis', ax=ax)
            ax.set_title('Model RMSE Comparison')
            ax.set_xlabel('Model')
            ax.set_ylabel('RMSE')
            ax.set_ylim(0, comparison_df['RMSE'].max() + 1)  # Ensure y-axis starts from 0 and ends beyond max RMSE value
            st.pyplot(fig)

            # MAE Bar Plot
            st.subheader("Model MAE Comparison")
            fig, ax = plt.subplots()
            sns.barplot(x=comparison_df.index, y='MAE', data=comparison_df, palette='viridis', ax=ax)
            ax.set_title('Model MAE Comparison')
            ax.set_xlabel('Model')
            ax.set_ylabel('MAE')
            ax.set_ylim(0, comparison_df['MAE'].max() + 1)  # Ensure y-axis starts from 0 and ends beyond max MAE value
            st.pyplot(fig)

else:
    st.info("Please upload a CSV file to proceed.")
