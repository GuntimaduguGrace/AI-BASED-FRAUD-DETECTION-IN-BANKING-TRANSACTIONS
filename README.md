#AI-BASED FRAUD DETECTION IN BANKING TRANSACTIONS:ENHANCING SECURITY AND EFFICIENCY BY USING ML AND PREDICTIVE ANALYTIC TECHNIQUES
## Project Description
This project detects fraudulent banking transactions using Machine Learning and Predictive Analytics. 
It enhances security and efficiency by identifying potentially fraudulent activities in real-time.
Users can upload their dataset, visualize transaction patterns, and compare multiple ML models for fraud detection.
## Dataset
- Dataset used: `online_fraud_detection.csv`
- Contains features like transaction type, amount, and fraud labels.
- Users can upload their own CSV to test the models in the app.
## Project Structure
data/ -> Sample CSV datasets

notebooks/ -> EDA and ML modeling

scripts/ -> Preprocessing, training, and evaluation scripts

models/ -> Trained machine learning models

app.py -> (Optional) Web app for interactive use

README.md -> Project documentation   

---

## Workflow / Features
The app allows users to explore the dataset, visualize trends, preprocess data, train multiple ML models, and compare their performance.

### 1. Data Upload
- Upload CSV file in the sidebar.
- Dataset info (shape, columns, sample data) is displayed.
  

### 2. Data Visualization
#### Univariate Analysis
- Transaction Type Distribution  
- Transaction Amount Distribution

#### Bivariate Analysis
- Transaction Type vs Fraud Distribution

### 3. Data Preprocessing
- Handle missing values and outliers  
- Encode categorical features  
- Normalize numerical features  

### 4. Model Training & Evaluation
- Implemented models:
  - **Logistic Regression**  
  - **Random Forest**  
  - **Support Vector Machine (SVM)**  

- For each model:
  - Confusion matrix visualization  
  - Classification report
### 5. Model Comparison
- Compared using multiple metrics:
  - Accuracy  
  - Root Mean Squared Error (RMSE)  
  - Mean Absolute Error (MAE)  
- Visual comparison of models’ performance  

---
## Technologies / Tools Used
- **Python** – Programming language  
- **Streamlit** – Interactive web app  
- **streamlit_lottie** – Add Lottie animations  
- **Pandas, NumPy** – Data handling & preprocessing  
- **Matplotlib, Seaborn** – Data visualization  
- **Scikit-learn** – ML models & evaluation metrics  
- **JSON** – Animation handling
## How to Run the Project
1. Clone the repository:  
   git clone https://github.com/yourusername/fraud-detection.git
2. Install dependencies:  
   pip install -r requirements.txt
3. Run the app:  
   streamlit run app.py
4.Upload online_fraud_detection.csv (or your dataset) and explore visualizations & model results.

## Results

The models were trained and evaluated on the dataset. Below are the performance metrics:

| Model                | Accuracy | RMSE   | MAE    |
|-----------------------|----------|--------|--------|
| Logistic Regression   | 0.9998   | 0.0155 | 0.0168 |
| Random Forest         | 0.9995   | 0.0232 | 0.1377 |
| Support Vector Machine (SVM) | 0.9970   | 0.0552 | 0.1186 |

### Key Insights
- **Logistic Regression** achieved the highest overall accuracy (99.98%), with the lowest RMSE and MAE.  
- **Random Forest** also performed well but had a higher MAE compared to Logistic Regression.  
- **SVM** had slightly lower accuracy and higher errors, making it less optimal for this dataset.  


##Future Enhancements

Integrate the model into a real-time transaction monitoring system for instant fraud alerts.

Explore deep learning models (e.g., LSTM, Autoencoders) for better anomaly detection.

Incorporate additional features such as customer behavior patterns and device/location data for improved accuracy.



