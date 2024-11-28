# Language Prediction App

This project is a simple web application that predicts the language of a given text using a Naive Bayes model. The app is built with Python and Streamlit for the frontend interface.

## Features
- Accepts user input text through a web interface.
- Uses a trained Naive Bayes classifier to predict the language of the input text.
- Displays the predicted language on the screen.

## Dataset
The app uses a dataset hosted on GitHub, which contains text samples and their corresponding language labels. The data is loaded dynamically from:
[Dataset Link](https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv)

## Installation and Setup
1. **Clone the repository**:
git clone https://github.com/SUMIT0110/Language_Detection.git
2. **Navigate to the project directory**:
cd language-prediction-app
3. **Install required dependencies**:
pip install -r requirements.txt
4. **Run the application**:
streamlit run app.py

## Requirements
- Python 3.7 or higher
- Required Python libraries:
- Streamlit
- Pandas
- NumPy
- Scikit-learn

## How It Works
1. **Data Preprocessing**:
- The dataset is loaded and split into features (`Text`) and labels (`language`).
- The text data is transformed into numerical format using `CountVectorizer`.

2. **Model Training**:
- A Multinomial Naive Bayes model is trained on the processed dataset.

3. **User Interaction**:
- The user inputs text through the Streamlit interface.
- The app preprocesses the text and uses the trained model to predict the language.

## Usage
1. Start the app using the command:
streamlit run app.py
2. Enter a piece of text into the text area.
3. Click the "Predict" button to see the predicted language.

## File Structure
- `app.py`: Main application code.
- `README.txt`: Documentation file.
- `requirements.txt`: List of dependencies required to run the app.

## Future Enhancements
- Add support for more languages.
- Improve model accuracy with additional datasets.
- Add functionality for confidence scores and analysis of predictions.

## Credits
This project is created using a dataset from Aman Kharwal's GitHub repository.
