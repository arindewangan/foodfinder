from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = pd.read_csv("food.csv")

# Vectorize ingredients
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
X = vectorizer.fit_transform(data['ingredients'])

# Deploy model
def predict_dishes(ingredients, threshold=0.6):
    # Vectorize input ingredients
    X_input = vectorizer.transform([ingredients])

    # Compute cosine similarity between input and all recipes
    similarities = cosine_similarity(X_input, X).flatten()

    # Get indices of recipes with similarity above threshold
    indices = [i for i, sim in enumerate(similarities) if sim > threshold]

    # Return the dish names of recipes with similarity above threshold
    return list(data.loc[indices]['dish'])


# Frontend
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def get():
    ingredients = request.form['ingredients']
    predicted_dishes = predict_dishes(ingredients)
    ing = ingredients.split(',')
    return render_template('result.html', predicted_dishes=predicted_dishes,ingredients=ingredients,ing=ing,len = len(ing))

if __name__ == '__main__':
    app.run(debug=True)
