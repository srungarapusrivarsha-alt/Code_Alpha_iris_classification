
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report

st.title("Iris Flower Classification")

data = pd.read_csv(
    r"C:\Users\sriva\Desktop\code_alpha\IRIS_CLASSIFICATION\Iris.csv"
)
x = data[['SepalLengthCm', 'SepalWidthCm',
          'PetalLengthCm', 'PetalWidthCm']]

y = data['Species']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42
)
model = SVC()
model.fit(x_train, y_train)
train_output = model.predict(x_train)
test_output = model.predict(x_test)

train_accuracy = accuracy_score(y_train, train_output)
train_precision = precision_score(y_train, train_output, average="macro")
train_recall = recall_score(y_train, train_output, average="macro")
train_f1_score = f1_score(y_train, train_output, average="macro")

test_accuracy = accuracy_score(y_test, test_output)
test_precision = precision_score(y_test, test_output, average="macro")
test_recall = recall_score(y_test, test_output, average="macro")
test_f1_score = f1_score(y_test, test_output, average="macro")

st.subheader("Predict Iris Flower")

sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, value=4.2)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.2)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, value=1.6)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, value=0.3)

if st.button("Predict"):
    prediction = model.predict([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    st.success("Predicted Species: " + prediction[0])

