from flask import Flask, request, jsonify
import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
from waitress import serve
import os

app = Flask(__name__)
CORS(app)
# Load data and models
DATA_PATH = './data/Data_Kegiatan_Volunteer_Realistic.xlsx'
DB_PATH = './database/users.db'

df = pd.read_excel(DATA_PATH)

def get_user_applied_projects(username):
    """Fetch the list of applied projects for a specific user from the database."""
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT applied_projects FROM users WHERE username = ?"
    cursor = conn.cursor()
    cursor.execute(query, (username,))
    applied_projects = cursor.fetchone()
    conn.close()
    if applied_projects:
        return applied_projects[0].split(',')  # Convert comma-separated string into a list of IDs
    return []

def calculate_cosine_similarity(df):
    """Calculate cosine similarity for a given DataFrame."""
    indonesian_stopwords = [
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "dengan", "pada", "oleh", "sebagai", 
    "ada", "karena", "kita", "akan", "saya", "kamu", "dia", "mereka", "kami", "tidak", "bisa", 
    "lebih", "lagi", "telah", "atau", "sudah", "jadi", "dalam", "luar", "hanya", "seperti", 
    "sangat", "semua", "banyak", "apa", "bagaimana", "mengapa", "kapan", "dapat", "harus", "setelah", 
    "sebelum", "kalau", "jika", "pun", "antara", "tetapi", "meskipun", "namun", "tanpa", "hingga", 
    "selama", "sejak", "bukan", "malah", "maupun", "oleh", "agar", "bahwa", "itu", "dapat", "belum", 
    "kemudian", "masih", "baru", "diri", "sekali", "selalu", "mau", "tentang", "atas", "dengan", 
    "cukup", "lalu", "bagi", "dahulu", "lain", "kembali", "sedang", "nya", "dapat", "begitu", 
    "bila", "sebuah", "kecil", "besar", "terhadap", "hingga", "semakin", "milik", "hal", "orang", 
    "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan", "sepuluh"
]
    tfidf = TfidfVectorizer(stop_words=indonesian_stopwords)
    
    # Calculate TF-IDF and cosine similarity
    tfidf_matrix = tfidf.fit_transform(df['Deskripsi Kegiatan'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    return cosine_sim

@app.route('/register', methods=['POST'])
def register():
    user_data = request.json
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute('INSERT INTO users (username, password, location, category, applied_projects) VALUES (?, ?, ?, ?, ?)',
                       (user_data['username'], user_data['password'], user_data['location'], user_data['category'], ''))
        conn.commit()
        return jsonify({'message': 'User registered successfully'}), 201
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username already exists'}), 400
    finally:
        conn.close()

@app.route('/login', methods=['POST'])
def login():
    credentials = request.json
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', 
                   (credentials['username'], credentials['password']))
    user = cursor.fetchone()
    conn.close()

    if user:
        return jsonify({
            'username': user[1],
            'location': user[3],
            'category': user[4],
            'applied_projects': user[5].split(',') if user[5] else []
        })
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/recommendations', methods=['GET'])
def recommendations():
    username = request.args.get('username')
    location = request.args.get('location')
    category = request.args.get('category')

    if not username:
        return jsonify({"error": "Username is required"}), 400

    print(f"Fetching recommendations for user: {username}")
    applied_projects = get_user_applied_projects(username)
    print(f"Applied projects for {username}: {applied_projects}")

    if applied_projects == ['']:  # New user without applied projects
        print("No applied projects found. Using exact matching.")
        recommendations = exact_matching_recommendations(location, category)
    else:  # User with applied projects
        applied_project_ids = [int(project_id) for project_id in applied_projects]
        filtered_df = df[df['id'].isin(applied_project_ids)]  # Filter only applied projects
        cosine_sim_full = calculate_cosine_similarity(df)
        cosine_sim_filtered = calculate_cosine_similarity(filtered_df)  # Calculate cosine similarity only for applied projects
        recommendations = cosine_similarity_recommendations(applied_projects, cosine_sim_full, cosine_sim_filtered, df)

    # Fallback to exact matching if no recommendations are found
    if not recommendations:
        recommendations = exact_matching_recommendations(location, category)


    return jsonify(recommendations)

def exact_matching_recommendations(location, category):
    """Provide recommendations based on exact matching."""
    recommendations = []
    
    # Safeguard against None or empty values
    location = str(location).strip().lower() if location else ''
    category = str(category).strip().lower() if category else ''


    # Exact matching on both location and category
    for _, row in df.iterrows():
        row_location = str(row['Lokasi (Kota, Provinsi)']).strip().lower()
        row_category = str(row['Kategori Kegiatan']).strip().lower()
        if location == row_location and category == row_category:
            recommendations.append(row.to_dict())
            if len(recommendations) >= 5:
                break

    # Fallback to location-only if no recommendations found
    if not recommendations:
        for _, row in df.iterrows():
            row_location = str(row['Lokasi (Kota, Provinsi)']).strip().lower()
            if location == row_location:
                recommendations.append(row.to_dict())
                if len(recommendations) >= 5:
                    break

    # Fallback to category-only if still no recommendations found
    if not recommendations:
        for _, row in df.iterrows():
            row_category = str(row['Kategori Kegiatan']).strip().lower()
            if category == row_category:
                recommendations.append(row.to_dict())
                if len(recommendations) >= 5:
                    break

    return recommendations

def cosine_similarity_recommendations(applied_projects, cosine_sim_full, cosine_sim_filtered, df): 
    """Provide recommendations based on cosine similarity from the entire DataFrame."""
    recommendations = []
    seen_projects = set(applied_projects)

    # Calculate cosine similarity on the entire dataframe
    sim_scores_full = cosine_sim_full.mean(axis=1)  # Average similarity across all projects in the full DataFrame
    sim_scores_filtered = cosine_sim_filtered.mean(axis=1)  # Average similarity across filtered projects
    
    for idx, project_id in enumerate(sim_scores_full):
        # Calculate difference in similarity scores between full and filtered
        similarity_diff = abs(sim_scores_full - sim_scores_filtered[idx])
        
        # Find the top 5 closest projects to the filtered project
        top_indices = similarity_diff.argsort()[:5]  # Select the top 5 closest indices
        
        for i in top_indices:
            original_id = df.iloc[i]['id']
            if original_id in seen_projects or original_id in applied_projects or sim_scores_full[i] <= 0:
                continue
            
            row = df[df['id'] == original_id].iloc[0]
            recommendations.append(row.to_dict())
            seen_projects.add(original_id)

            # Stop if we have enough recommendations
            if len(recommendations) >= 5:
                break

        if len(recommendations) >= 5:
            break

    print(f"Recommendations after cosine similarity (excluding applied projects): {recommendations}")
    return recommendations

    

@app.route('/apply', methods=['POST'])
def apply():
    data = request.json
    username = data['username']
    project_id = data['project_id']

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT applied_projects FROM users WHERE username = ?', (username,))
    applied_projects = cursor.fetchone()[0]
    applied_projects = applied_projects.split(',') if applied_projects else []
    applied_projects.append(str(project_id))

    cursor.execute('UPDATE users SET applied_projects = ? WHERE username = ?',
                   (','.join(applied_projects), username))
    conn.commit()
    conn.close()
    print(f"Applying for project ID: {project_id}")
    return jsonify({'message': 'Application successful'})

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    query = query.strip().lower()

    results = df[
        df['Nama Kegiatan'].str.lower() == query |
        df['Kategori Kegiatan'].str.lower() == query |
        df['Lokasi (Kota, Provinsi)'].str.lower() == query
    ].to_dict(orient='records')

    if not results:
        results = [{'message': 'No projects found matching your query.'}]
    
    return jsonify(results)

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
