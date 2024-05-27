import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load the data
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv")

# Preprocess the data
x = np.array(data["Text"])
y = np.array(data["language"])
cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Streamlit app
def app():
    st.title("Language Prediction App")
    st.write("Enter text to predict the language:")

    user_input = st.text_area("", height=200)
    
    if st.button("Predict"):
        if user_input:
            data = cv.transform([user_input]).toarray()
            output = model.predict(data)[0]
            st.success(f"The predicted language is: {output}")
        else:
            st.warning("Please enter some text to predict the language.")

if __name__ == "__main__":
    app()