from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Dataset 1: movie_id,movie_name,year,genre,overview,director,cast
df1 = pd.read_csv("dataset2.csv").fillna('')
df1['combined_features'] = df1['genre'] + ' ' + df1['director'] + ' ' + df1['cast']
df1['movie_name'] = df1['movie_name'].str.strip()
vectorizer1 = TfidfVectorizer(stop_words='english')
tfidf_matrix1 = vectorizer1.fit_transform(df1['combined_features'])
cosine_sim1 = cosine_similarity(tfidf_matrix1, tfidf_matrix1)
indices1 = pd.Series(df1.index, index=df1['movie_name'].str.lower())

# Dataset 2: imdbId,title,releaseYear,releaseDate,genre,writers,actors,directors,sequel,hitFlop
df2 = pd.read_csv("dataset1.csv").fillna('')
df2['combined_features'] = df2['genre'] + ' ' + df2['writers'] + ' ' + df2['actors'] + ' ' + df2['directors']
df2['title'] = df2['title'].str.strip()
vectorizer2 = TfidfVectorizer(stop_words='english')
tfidf_matrix2 = vectorizer2.fit_transform(df2['combined_features'])
cosine_sim2 = cosine_similarity(tfidf_matrix2, tfidf_matrix2)
indices2 = pd.Series(df2.index, index=df2['title'].str.lower())

# Dataset 3: Name,Year,Duration,Genre,Rating,Votes,Director,Actor 1,Actor 2,Actor 3
df3 = pd.read_csv("dataset3.csv").fillna('')
df3['combined_features'] = df3['Genre'] + ' ' + df3['Director'] + ' ' + df3['Actor 1'] + ' ' + df3['Actor 2'] + ' ' + df3['Actor 3']
df3['Name'] = df3['Name'].str.strip()
vectorizer3 = TfidfVectorizer(stop_words='english')
tfidf_matrix3 = vectorizer3.fit_transform(df3['combined_features'])
cosine_sim3 = cosine_similarity(tfidf_matrix3, tfidf_matrix3)
indices3 = pd.Series(df3.index, index=df3['Name'].str.lower())

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    selected_dataset = "1"
    
    if request.method == 'POST':
        title = request.form['title'].strip().lower()
        selected_dataset = request.form['dataset']
        
        if selected_dataset == "1":
            if title in indices1:
                idx = indices1[title]
                sim_scores = list(enumerate(cosine_sim1[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
                movie_indices = [i[0] for i in sim_scores]
                recommendations = df1['movie_name'].iloc[movie_indices].tolist()
            else:
                recommendations = ["Movie not found in Dataset 1."]
                
        elif selected_dataset == "2":
            if title in indices2:
                idx = indices2[title]
                sim_scores = list(enumerate(cosine_sim2[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
                movie_indices = [i[0] for i in sim_scores]
                recommendations = df2['title'].iloc[movie_indices].tolist()
            else:
                recommendations = ["Movie not found in Dataset 2."]
                
        elif selected_dataset == "3":
            if title in indices3:
                idx = indices3[title]
                sim_scores = list(enumerate(cosine_sim3[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
                movie_indices = [i[0] for i in sim_scores]
                recommendations = df3['Name'].iloc[movie_indices].tolist()
            else:
                recommendations = ["Movie not found in Dataset 3."]
                
    return render_template("index.html", recommendations=recommendations, selected_dataset=selected_dataset)

if __name__ == '__main__':
    app.run(debug=True)
