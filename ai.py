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

ingredients = 'Maida, corn flour, baking soda, vinegar, curd, water, turmeric, saffron, cardamom'
predicted_dish = predict_dish(ingredients)
print(predicted_dish)

