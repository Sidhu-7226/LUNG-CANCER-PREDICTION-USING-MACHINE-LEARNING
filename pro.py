import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load and preprocess data
data = pd.read_csv('C:\\Users\\golla\\Downloads\\survey-lung-cancer.csv')

# Strip any spaces and standardize column names
data.columns = data.columns.str.strip().str.replace(' ', '_').str.upper()

# Convert categorical variables to numerical values
label_encoders = {
    'GENDER': LabelEncoder(),
    'SMOKING': LabelEncoder(),
    'YELLOW_FINGERS': LabelEncoder(),
    'ANXIETY': LabelEncoder(),
    'PEER_PRESSURE': LabelEncoder(),
    'CHRONIC_DISEASE': LabelEncoder(),
    'FATIGUE': LabelEncoder(),
    'ALLERGY': LabelEncoder(),
    'WHEEZING': LabelEncoder(),
    'ALCOHOL_CONSUMING': LabelEncoder(),
    'COUGHING': LabelEncoder(),
    'SHORTNESS_OF_BREATH': LabelEncoder(),
    'SWALLOWING_DIFFICULTY': LabelEncoder(),
    'CHEST_PAIN': LabelEncoder(),
    'LUNG_CANCER': LabelEncoder()
}

for column, le in label_encoders.items():
    data[column] = le.fit_transform(data[column])

# Separate features and target variable
X = data.drop('LUNG_CANCER', axis=1)
y = data['LUNG_CANCER']

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Streamlit application
st.title("Lung Cancer Prediction")

st.sidebar.header("Patient Data")
gender = st.sidebar.selectbox("Gender", ("M", "F"))
age = st.sidebar.slider("Age", 1, 100, 25)
smoking = st.sidebar.selectbox("Smoking", ("NO", "YES"))
yellow_fingers = st.sidebar.selectbox("Yellow Fingers", ("NO", "YES"))
anxiety = st.sidebar.selectbox("Anxiety", ("NO", "YES"))
peer_pressure = st.sidebar.selectbox("Peer Pressure", ("NO", "YES"))
chronic_disease = st.sidebar.selectbox("Chronic Disease", ("NO", "YES"))
fatigue = st.sidebar.selectbox("Fatigue", ("NO", "YES"))
allergy = st.sidebar.selectbox("Allergy", ("NO", "YES"))
wheezing = st.sidebar.selectbox("Wheezing", ("NO", "YES"))
alcohol_consuming = st.sidebar.selectbox("Alcohol Consuming", ("NO", "YES"))
coughing = st.sidebar.selectbox("Coughing", ("NO", "YES"))
shortness_of_breath = st.sidebar.selectbox("Shortness of Breath", ("NO", "YES"))
swallowing_difficulty = st.sidebar.selectbox("Swallowing Difficulty", ("NO", "YES"))
chest_pain = st.sidebar.selectbox("Chest Pain", ("NO", "YES"))

new_data = {
    'GENDER': gender,
    'AGE': age,
    'SMOKING': smoking,
    'YELLOW_FINGERS': yellow_fingers,
    'ANXIETY': anxiety,
    'PEER_PRESSURE': peer_pressure,
    'CHRONIC_DISEASE': chronic_disease,
    'FATIGUE': fatigue,
    'ALLERGY': allergy,
    'WHEEZING': wheezing,
    'ALCOHOL_CONSUMING': alcohol_consuming,
    'COUGHING': coughing,
    'SHORTNESS_OF_BREATH': shortness_of_breath,
    'SWALLOWING_DIFFICULTY': swallowing_difficulty,
    'CHEST_PAIN': chest_pain
}

# Convert the input to a DataFrame
input_df = pd.DataFrame([new_data])

# Encode categorical features using the same encoder fitted on training data
for column, le in label_encoders.items():
    if column in input_df.columns:
        if input_df[column].iloc[0] not in le.classes_:
            st.error(f"Unexpected value '{input_df[column].iloc[0]}' for {column}.")
        else:
            input_df[column] = le.transform(input_df[column])

# Normalize the data using the same scaler fitted on training data
input_df = scaler.transform(input_df)

# Make prediction
prediction = model.predict(input_df)
prediction_label = label_encoders['LUNG_CANCER'].inverse_transform(prediction)[0]

# Display prediction with colors
if prediction_label == 'YES':
    st.markdown(f"<h1 style='text-align: center; color: red;'>Lung Cancer Prediction: {prediction_label}</h1>", unsafe_allow_html=True)
else:
    st.markdown(f"<h1 style='text-align: center; color: green;'>Lung Cancer Prediction: {prediction_label}</h1>", unsafe_allow_html=True)

# Display accuracy and other metrics
st.write(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
st.write('Classification Report:')
st.text(classification_report(y_test, y_pred))
st.write('Confusion Matrix:')
st.write(confusion_matrix(y_test, y_pred))
