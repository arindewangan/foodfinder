from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
recipes = pd.read_csv('food.csv')

# Vectorize the ingredients
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(recipes['ingredients'])

# Train the model
y = recipes['dish']
model = RandomForestClassifier()
model.fit(X, y)

# Predict food dishes based on input ingredients
def predict_dish(ingredients):
    X_test = vectorizer.transform([ingredients])
    y_pred = model.predict(X_test)
    return y_pred[0]


# Frontend
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def get():
    ingredients = request.form['ingredients']
    predicted_dish = predict_dish(ingredients)
    ing = ingredients.split(',')
    return render_template('result.html', predicted_dish=predicted_dish,ingredients=ingredients,ing=ing,len = len(ing))

if __name__ == '__main__':
    app.run(debug=True)
