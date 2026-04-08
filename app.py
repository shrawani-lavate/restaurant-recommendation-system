import pandas as pd
import numpy as np
import re
from flask import Flask, render_template, request, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load and Preprocess Data
def load_data():
    df = pd.read_csv('restaurant1.csv')
    df = df.drop_duplicates(subset='name') # Ensure unique restaurants
    df = df.reset_index(drop=True)
    
    # Add dummy data to match the screenshot UI
    np.random.seed(42)
    df['Mean Rating'] = np.round(np.random.uniform(3.0, 5.0, size=len(df)), 2)
    # Some costs are high (900.0), some are low (1.2) in the screenshot
    df['cost'] = np.random.choice([1.2, 1.4, 1.5, 1.6, 2.1, 250.0, 300.0, 400.0, 500.0, 800.0, 900.0], size=len(df))
    
    return df

def clean_reviews(text):
    if pd.isna(text):
        return ""
    # Removing Zomato specific rating patterns and special characters
    text = re.sub(r'RATED\n|Rated \d\.\d|rated \d\.\d', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text

# Global Variables for ML
df = load_data()
df['cleaned_reviews'] = df['reviews_list'].apply(clean_reviews)
# Combine reviews with cuisines for better matching
df['features'] = df['cleaned_reviews'] + " " + df['cuisines'].fillna('')

#The Model code
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_restaurants(name, cosine_sim=cosine_sim):
    try:
        # Get index of the restaurant that matches the name
        idx = df[df['name'].str.lower() == name.lower()].index[0]
        
        # Get pairwise similarity scores of all restaurants with that restaurant
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort restaurants based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get scores of the 10 most similar restaurants (excluding self)
        sim_scores = sim_scores[1:11]
        
        # Get restaurant indices
        restaurant_indices = [i[0] for i in sim_scores]
        
        # Return the top 10 most similar restaurants
        return df.iloc[restaurant_indices][['name', 'cuisines', 'Mean Rating', 'cost']]
    except Exception as e:
        print(f"Error in recommendation: {e}")
        return pd.DataFrame()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/extractor')
def extractor():
    restaurant_names = df['name'].tolist()
    return render_template('extractor.html', restaurants=restaurant_names)

@app.route('/keywords', methods=['POST'])
def keywords():
    restaurant_name = request.form.get('restaurant_name')
    if not restaurant_name:
        return redirect(url_for('extractor'))
    
    recommendations = recommend_restaurants(restaurant_name)
    
    if recommendations.empty:
        return render_template('keywords.html', name=restaurant_name, table=None)
    
    # Prepare the table for display exactly like the screenshot
    # Rename 'name' to an empty string to match the screenshot header
    recommendations = recommendations.rename(columns={'name': ''})
    
    # Convert to HTML table with custom classes for styling
    html_table = recommendations.to_html(classes='recommend-table', index=False, border=1)
    
    return render_template('keywords.html', name=restaurant_name, table=html_table)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
