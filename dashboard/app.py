import dash
from dash import dcc, html, Input, Output, State, ALL
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
import time
import random
import networkx as nx
from dash.exceptions import PreventUpdate


#load data
df = pd.read_csv('../data/songs.csv')
df = df.dropna(subset=['Name', 'Release'])
df = df.rename(columns={'Valeance': 'Valence'})

#use all genres
df_filtered = df.copy()
all_genres = sorted(df_filtered['Genre'].unique())

#audio features
audio_features = ['Danceability', 'Energy', 'Loudness', 'Speechiness', 
                  'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']

#dictionary to store trained models
trained_models = {}

#store selected genres for training
selected_training_genres = []

trained_models = {}

selected_training_genres = []

#try to load pre-trained hierarchical model
try:
    #load lvl 1 (parent genre classifier)
    with open('../models/hierarchical/parent_model.pkl', 'rb') as f:
        parent_model = pickle.load(f)
    with open('../models/hierarchical/parent_scaler.pkl', 'rb') as f:
        parent_scaler = pickle.load(f)
    with open('../models/hierarchical/parent_encoder.pkl', 'rb') as f:
        parent_encoder = pickle.load(f)
    
    #load metadata to get list of lvl 2 models
    with open('../models/hierarchical/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    # load all lvl 2 models
    level2_models = {}
    level2_scalers = {}
    level2_encoders = {}
    
    for parent_genre in metadata['level2_parents']:
        safe_name = parent_genre.replace(' ', '_').replace('&', 'and').replace('/', '_')
        try:
            with open(f'../models/hierarchical/{safe_name}_model.pkl', 'rb') as f:
                level2_models[parent_genre] = pickle.load(f)
            with open(f'../models/hierarchical/{safe_name}_scaler.pkl', 'rb') as f:
                level2_scalers[parent_genre] = pickle.load(f)
            with open(f'../models/hierarchical/{safe_name}_encoder.pkl', 'rb') as f:
                level2_encoders[parent_genre] = pickle.load(f)
        except FileNotFoundError:
            print(f" Level 2 model for '{parent_genre}' not found, skipping...")
            continue
    
    # store hierarchical model
    trained_models['Hierarchical Model (Pre-trained)'] = {
        'type': 'hierarchical',
        'parent_model': parent_model,
        'parent_scaler': parent_scaler,
        'parent_encoder': parent_encoder,
        'level2_models': level2_models,
        'level2_scalers': level2_scalers,
        'level2_encoders': level2_encoders,
        'feature_names': metadata['feature_names'],
        'metadata': metadata
    }
    
    hierarchical_loaded = True
    print(f" Hierarchical model loaded successfully!")
    print(f"   - {len(parent_encoder.classes_)} parent genres")
    print(f"   - {len(level2_models)} specialized Level 2 classifiers")
    print(f"   - Total accuracy: {metadata['hierarchical_accuracy']*100:.2f}%")
    
except FileNotFoundError:
    hierarchical_loaded = False
    print("⚠️ Hierarchical models not found.")

#try to load legacy flat model (backwards compatibility)
try:
    with open('../models/genre_classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('../models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('../models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    trained_models['Legacy Flat Model'] = {
        'type': 'flat',
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder
    }
    print(" Legacy flat model loaded successfully!")
    
except FileNotFoundError:
    print(" Legacy flat models not found.")

if not trained_models:
    print(" No pre-trained models found. You can train models using the interface!")


#initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Music Genre Classification Dashboard"

#define colors 
colors = {
    'background': '#0a0e27',
    'card': '#1a1f3a',
    'text': '#ffffff',
    'accent': '#00d9ff',
    'secondary': '#ff006e',
    'tertiary': '#7209b7',
    'success': '#06ffa5',
    'node_default': '#4361ee',
    'node_selected': '#f72585',
    'edge': 'rgba(99, 102, 241, 0.15)'
}

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Dropdown styling for dark mode */
            .Select-control {
                background-color: #0a0e27 !important;
                border-color: #00d9ff !important;
            }
            .Select-menu-outer {
                background-color: #1a1f3a !important;
                border-color: #00d9ff !important;
            }
            .Select-option {
                background-color: #1a1f3a !important;
                color: #ffffff !important;
            }
            .Select-option:hover {
                background-color: #00d9ff !important;
                color: #0a0e27 !important;
            }
            .Select-value-label {
                color: #ffffff !important;
            }
            .Select-placeholder {
                color: #ffffff !important;
                opacity: 0.6;
            }
            .Select-input > input {
                color: #ffffff !important;
            }
            /* Multi-select tags */
            .Select-value {
                background-color: #00d9ff !important;
                color: #0a0e27 !important;
                border-color: #00d9ff !important;
            }
            .Select-value-icon {
                border-color: #0a0e27 !important;
            }
            .Select-value-icon:hover {
                background-color: #0099cc !important;
                color: #ffffff !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

#precompute genre network data for faster loading
def precompute_genre_network():
    """Pre-compute the 3D network layout for all genres - OPTIMIZED VERSION"""
    print(" Step 1: Computing genre features...")
    genre_counts = df['Genre'].value_counts()
    top_genres = genre_counts.index.tolist()[:5000]  # all 5000 genres
    
    # compute feature centroids for each genre
    genre_features = {}
    for genre in top_genres:
        genre_features[genre] = df[df['Genre'] == genre][audio_features].mean().values
    
    feature_matrix = np.array([genre_features[g] for g in top_genres])
    
    print(" Step 2: Reducing to 3D using PCA (fast method)...")
    # use PCA instead of t-SNE (faster for 5000 points)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3, random_state=42)
    positions_3d = pca.fit_transform(feature_matrix)
    
    #normalize positions for better visualization
    positions_3d = (positions_3d - positions_3d.mean(axis=0)) / positions_3d.std(axis=0)
    
    # create position dictionary
    pos = {genre: positions_3d[i] for i, genre in enumerate(top_genres)}
    
    print(" Step 3: Building network connections...")
    #build lightweight graph with selective edges
    G = nx.Graph()
    for genre in top_genres:
        G.add_node(genre)
    
    #optimized edge creation - only compute for first 2000 genres to save time
    #users can still select any of the 5000 genres
    genre_list = top_genres[:2000]  # Limit edge computation
    for i, g1 in enumerate(genre_list):
        if i % 200 == 0:
            print(f"  Processing connections... {i}/{len(genre_list)}")
        
        vec1 = genre_features[g1]
        similarities = []
        
        #only compare with nearby genres (next 500)
        for j in range(i+1, min(i+500, len(genre_list))):
            g2 = genre_list[j]
            vec2 = genre_features[g2]
            #cosine similarity
            sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
            if sim > 0.85:  #only strong similarities
                similarities.append((g2, sim))
        
        #only keep top 3 connections per node
        for g2, sim in sorted(similarities, key=lambda x: -x[1])[:3]:
            G.add_edge(g1, g2, weight=sim)
    
    print(" Network computation complete!")
    return G, pos, genre_counts, top_genres

#precompute on startup
print(" Computing genre network ...")
GENRE_GRAPH, GENRE_POSITIONS, GENRE_COUNTS, TOP_GENRES = precompute_genre_network()
print(f" Network ready with {len(TOP_GENRES)} genres!")

#precompute genre statistics for instant visualization
print(" Pre-computing genre statistics for visualization...")
GENRE_STATS = {}
for genre in TOP_GENRES:
    genre_df = df[df['Genre'] == genre]
    GENRE_STATS[genre] = {
        feat: genre_df[feat].mean() for feat in audio_features
    }
print(" Genre statistics ready!")

#app layout
app.layout = html.Div(style={'backgroundColor': colors['background'], 'minHeight': '100vh', 'padding': '20px'}, children=[
    
    # header
    html.Div([
        html.H1(" Music Genre Classification Dashboard", 
                style={'textAlign': 'center', 'color': colors['accent'], 'marginBottom': '10px',
                       'fontSize': '48px', 'fontWeight': '900', 'letterSpacing': '2px'}),
        html.P(f"Analyzing {len(df_filtered):,} songs across {len(all_genres)} genres using machine learning",
               style={'textAlign': 'center', 'color': colors['text'], 'fontSize': '18px', 'opacity': '0.8'})
    ]),
    
    html.Hr(style={'borderColor': colors['accent'], 'opacity': '0.3', 'marginBottom': '40px'}),
    
    #genre selection section
    html.Div([
        html.H2(" Genre Universe Explorer", 
                style={'color': colors['accent'], 'marginBottom': '20px', 'fontSize': '32px'}),
        html.P("Select genres for training using one of these methods:", 
               style={'color': colors['text'], 'fontSize': '16px', 'marginBottom': '30px'}),
        
        # selection method tabs 
        html.Div([
            html.Button(' Random Selection', id='tab-random', n_clicks=0,
                       style={'backgroundColor': colors['accent'], 'color': colors['background'], 
                              'padding': '15px 30px', 'fontSize': '16px', 'fontWeight': 'bold',
                              'border': 'none', 'borderRadius': '10px 10px 0 0', 'cursor': 'pointer',
                              'marginRight': '5px'}),
            html.Button(' Visual Explorer', id='tab-visual', n_clicks=0,
                       style={'backgroundColor': colors['card'], 'color': colors['text'], 
                              'padding': '15px 30px', 'fontSize': '16px', 'fontWeight': 'bold',
                              'border': 'none', 'borderRadius': '10px 10px 0 0', 'cursor': 'pointer'}),
        ], style={'marginBottom': '0'}),
        
        #content area
        html.Div(id='genre-selection-content', 
                style={'backgroundColor': colors['card'], 'padding': '30px', 
                       'borderRadius': '0 10px 10px 10px', 'minHeight': '400px'}),
        
        #selected genres display
        html.Div([
            html.H4(" Selected Genres:", style={'color': colors['accent'], 'marginTop': '20px'}),
            html.Div(id='selected-genres-display', 
                    style={'color': colors['text'], 'fontSize': '14px', 'marginTop': '10px'})
        ]),
        
    ], style={'marginBottom': '40px'}),
    
    # store components for genre selection
    dcc.Store(id='active-tab', data='random'),
    dcc.Store(id='selected-genres-store', data=[]),
    
    # model training section
    html.Div([
        html.H3(" Train Models", style={'color': colors['accent'], 'fontSize': '28px'}),
        html.P("Select and train machine learning models on your chosen genres:", 
               style={'color': colors['text']}),
        
        dcc.Checklist(
            id='model-selector',
            options=[
                {'label': ' Random Forest ', 'value': 'Random Forest'},
                {'label': ' Logistic Regression ', 'value': 'Logistic Regression'},
                {'label': ' Support Vector Machine ', 'value': 'SVM'}
            ],
            value=['Random Forest'],
            style={'color': colors['text'], 'fontSize': '16px'},
            labelStyle={'display': 'block', 'marginBottom': '15px', 'cursor': 'pointer'}
        ),
        
        html.Button(' Train Selected Models', id='train-button', n_clicks=0,
                   style={'backgroundColor': colors['secondary'], 'color': colors['text'], 
                          'padding': '15px 35px', 'fontSize': '18px', 'fontWeight': 'bold',
                          'border': 'none', 'borderRadius': '10px', 'cursor': 'pointer',
                          'marginTop': '20px', 'boxShadow': '0 4px 15px rgba(255, 0, 110, 0.4)'}),
        
        html.Div(id='training-output', style={'marginTop': '20px'})
        
    ], style={'backgroundColor': colors['card'], 'padding': '30px', 'borderRadius': '15px', 
              'marginBottom': '40px', 'boxShadow': '0 4px 20px rgba(0, 217, 255, 0.1)'}),
    
    #filters section
    html.Div([
        html.H3(" Visualization Filters", style={'color': colors['accent'], 'fontSize': '28px'}),
        
        html.Div([
            html.Label("Filter by Genre:", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '16px'}),
            dcc.Dropdown(
                id='genre-filter',
                options=[{'label': genre, 'value': genre} for genre in all_genres],
                value=all_genres[:10],
                multi=True,
                style={'backgroundColor': colors['background'], 'color': colors['text']}
            ),
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.Label("Select Features to Compare:", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '16px'}),
            dcc.Dropdown(
                id='feature-selector',
                options=[{'label': feat, 'value': feat} for feat in audio_features],
                value=['Danceability', 'Energy'],
                multi=True,
                style={'backgroundColor': colors['background'], 'color': colors['text']}
            ),
        ], style={'marginBottom': '20px'}),
        
    ], style={'backgroundColor': colors['card'], 'padding': '30px', 'borderRadius': '15px', 
              'marginBottom': '40px', 'boxShadow': '0 4px 20px rgba(0, 217, 255, 0.1)'}),
    
    #statistics cards
    html.Div(id='stats-cards', style={'marginBottom': '40px'}),
    
    # visualizations grid
    html.Div([
        
        html.Div([
            html.Div([
                dcc.Graph(id='genre-distribution')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
            
            html.Div([
                dcc.Graph(id='feature-correlation')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
        ]),
        
        
        html.Div([
            html.Div([
                dcc.Graph(id='scatter-plot')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
            
            html.Div([
                dcc.Graph(id='box-plot')
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
        ]),
        
        
        html.Div([
            dcc.Graph(id='radar-chart')
        ], style={'padding': '10px'}),
    ]),
    
    #song search section
    html.Div([
        html.H3(" Song Search & Analysis", style={'color': colors['accent'], 'fontSize': '28px'}),
        html.Label("Search for a Song:", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '16px'}),
        dcc.Input(
            id='song-search',
            type='text',
            placeholder='Type song name or artist...',
            style={
                'width': '100%', 
                'padding': '15px', 
                'backgroundColor': colors['background'],
                'color': colors['text'],
                'border': f'2px solid {colors["accent"]}',
                'borderRadius': '10px',
                'fontSize': '16px'
            }
        ),
        html.Div(id='search-results', style={'marginTop': '20px'})
    ], style={'backgroundColor': colors['card'], 'padding': '30px', 'borderRadius': '15px', 
              'marginTop': '40px', 'boxShadow': '0 4px 20px rgba(0, 217, 255, 0.1)'}),

    #manual prediction section
    html.Div([
        html.Hr(style={'borderColor': colors['accent'], 'opacity': '0.3', 'margin': '40px 0'}),
        html.H3(" Manual Genre Prediction Tool", style={'color': colors['accent'], 'fontSize': '28px'}),
        html.P("Adjust the sliders to predict the genre based on audio features:", 
               style={'color': colors['text'], 'fontSize': '16px'}),
        
        html.Div([
            html.Label("Select Model:", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '16px'}),
            dcc.Dropdown(
                id='prediction-model-selector',
                options=[],
                value=None,
                style={'backgroundColor': colors['background'], 'color': colors['text'], 'marginBottom': '30px'}
            ),
        ]),
        
        html.Div([
            html.Div([
                html.Label(f"{feat}:", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '14px'}),
                dcc.Slider(
                    id=f'slider-{feat}',
                    min=-60 if feat == 'Loudness' else (0 if feat != 'Tempo' else 0),
                    max=0 if feat == 'Loudness' else (1 if feat != 'Tempo' else 250),
                    value=-10 if feat == 'Loudness' else (120 if feat == 'Tempo' else 0.5),
                    marks={i/10: str(i/10) for i in range(0, 11)} if feat not in ['Tempo', 'Loudness'] else 
                           ({i*50: str(i*50) for i in range(0, 6)} if feat == 'Tempo' else {-60: '-60', -40: '-40', -20: '-20', 0: '0'}),
                    step=0.01,
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'marginBottom': '20px'}) for feat in audio_features
        ], style={'backgroundColor': colors['background'], 'padding': '25px', 'borderRadius': '10px'}),
        
        html.Div([
            html.Button(' Predict Genre', id='predict-button', n_clicks=0,
                       style={'backgroundColor': colors['accent'], 'color': colors['background'], 
                              'padding': '15px 40px', 'fontSize': '18px', 'fontWeight': 'bold',
                              'border': 'none', 'borderRadius': '10px', 'cursor': 'pointer',
                              'marginTop': '20px', 'boxShadow': '0 4px 15px rgba(0, 217, 255, 0.4)'}),
        ], style={'textAlign': 'center'}),
        
        html.Div(id='prediction-output', style={'marginTop': '30px', 'textAlign': 'center'})
        
    ], style={'marginTop': '40px'}),
    
    #footer
    html.Hr(style={'borderColor': colors['accent'], 'opacity': '0.3', 'marginTop': '60px'}),
    html.P("Created for Data Visualization & Analysis Final Project | Music Genre Classification",
           style={'textAlign': 'center', 'color': colors['text'], 'fontSize': '14px', 'opacity': '0.6'})
])

def predict_with_hierarchical_model(features, model_data):
    """
    Make predictions using the 2-level hierarchical model.
    
    Args:
        features: numpy array of audio features (should match training features)
        model_data: dictionary containing hierarchical model components
    
    Returns:
        tuple: (predicted_genre, confidence, top_3_predictions)
    """
    #extract model components
    parent_model = model_data['parent_model']
    parent_scaler = model_data['parent_scaler']
    parent_encoder = model_data['parent_encoder']
    level2_models = model_data['level2_models']
    level2_scalers = model_data['level2_scalers']
    level2_encoders = model_data['level2_encoders']
    
    #step 1:predict parent genre
    features_scaled = parent_scaler.transform(features)
    parent_pred_idx = parent_model.predict(features_scaled)[0]
    parent_genre = parent_encoder.inverse_transform([parent_pred_idx])[0]
    
    #get parent confidence if available
    if hasattr(parent_model, 'predict_proba'):
        parent_proba = parent_model.predict_proba(features_scaled)[0]
        parent_confidence = parent_proba[parent_pred_idx]
    else:
        parent_confidence = None
    
    # step 2:predict subgenre using lvl 2 model
    if parent_genre in level2_models:
        #scale features for lvl 2 model
        features_l2_scaled = level2_scalers[parent_genre].transform(features)
        
        #predict subgenre
        subgenre_pred_idx = level2_models[parent_genre].predict(features_l2_scaled)[0]
        predicted_genre = level2_encoders[parent_genre].inverse_transform([subgenre_pred_idx])[0]
        
        #get subgenre probabilities
        if hasattr(level2_models[parent_genre], 'predict_proba'):
            subgenre_proba = level2_models[parent_genre].predict_proba(features_l2_scaled)[0]
            
            #get top 3 subgenres within this parent
            top_3_idx = np.argsort(subgenre_proba)[-3:][::-1]
            top_3_genres = level2_encoders[parent_genre].inverse_transform(top_3_idx)
            top_3_probs = subgenre_proba[top_3_idx]
            
            #adjust confidence by parent confidence if available
            if parent_confidence is not None:
                confidence = subgenre_proba[subgenre_pred_idx] * parent_confidence
                top_3_probs = top_3_probs * parent_confidence
            else:
                confidence = subgenre_proba[subgenre_pred_idx]
        else:
            confidence = parent_confidence if parent_confidence is not None else 1.0
            top_3_genres = [predicted_genre]
            top_3_probs = [confidence]
    else:
        #no lvl 2 model for this parent -use parent genre as prediction
        predicted_genre = parent_genre
        confidence = parent_confidence if parent_confidence is not None else 1.0
        top_3_genres = [parent_genre]
        top_3_probs = [confidence]
    
    return predicted_genre, confidence, list(zip(top_3_genres, top_3_probs))

# callbacks

#tab switching 
@app.callback(
    [Output('tab-random', 'style'),
     Output('tab-visual', 'style'),
     Output('active-tab', 'data')],
    [Input('tab-random', 'n_clicks'),
     Input('tab-visual', 'n_clicks')],
    prevent_initial_call=True
)
def switch_tab(random_clicks, visual_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        active = 'random'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        active = button_id.replace('tab-', '')
    
    base_style = {'padding': '15px 30px', 'fontSize': '16px', 'fontWeight': 'bold',
                  'border': 'none', 'borderRadius': '10px 10px 0 0', 'cursor': 'pointer'}
    
    active_style = {**base_style, 'backgroundColor': colors['accent'], 'color': colors['background']}
    inactive_style = {**base_style, 'backgroundColor': colors['card'], 'color': colors['text']}
    
    return (
        active_style if active == 'random' else inactive_style,
        active_style if active == 'visual' else inactive_style,
        active
    )


#genre selection content
@app.callback(
    Output('genre-selection-content', 'children'),
    Input('active-tab', 'data')
)
def update_genre_selection_content(active_tab):
    if active_tab == 'random':
        return html.Div([
            html.H4(" Random Genre Selection", style={'color': colors['accent'], 'marginBottom': '20px'}),
            html.P("Randomly select a specified number of genres from the entire dataset:", 
                   style={'color': colors['text'], 'marginBottom': '20px'}),
            
            html.Label(f"Number of Genres (out of {len(all_genres)} total):", 
                      style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '16px'}),
            dcc.Slider(
                id='random-genre-count',
                min=5,
                max=min(100, len(all_genres)),
                value=20,
                marks={i: str(i) for i in [5, 10, 20, 30, 50, 75, 100] if i <= len(all_genres)},
                step=1,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            
            html.Button(' Generate Random Selection', id='generate-random-btn', n_clicks=0,
                       style={'backgroundColor': colors['success'], 'color': colors['background'], 
                              'padding': '12px 25px', 'fontSize': '16px', 'fontWeight': 'bold',
                              'border': 'none', 'borderRadius': '8px', 'cursor': 'pointer',
                              'marginTop': '30px'}),
        ])
    
    else:  # visual 
        return html.Div([
            html.H4(" Visual Genre Explorer - Blindsight Inspired", 
                   style={'color': colors['accent'], 'marginBottom': '20px'}),
            html.P(f"Explore all {len(TOP_GENRES)} genres in an interactive 3D universe. Similar genres cluster together based on audio features.", 
                   style={'color': colors['text'], 'marginBottom': '20px'}),
            
            html.Div([
                html.Label(" Color Genres by:", style={'color': colors['text'], 'fontWeight': 'bold', 'marginRight': '15px'}),
                dcc.Dropdown(
                    id='color-by-feature',
                    options=[{'label': feat, 'value': feat} for feat in audio_features],
                    value='Energy',
                    style={'width': '200px', 'display': 'inline-block', 'backgroundColor': colors['background']}
                ),
                html.Label(" Size Genres by:", style={'color': colors['text'], 'fontWeight': 'bold', 'marginLeft': '30px', 'marginRight': '15px'}),
                dcc.Dropdown(
                    id='size-by-metric',
                    options=[
                        {'label': 'Song Count', 'value': 'count'},
                        {'label': 'Danceability', 'value': 'Danceability'},
                        {'label': 'Energy', 'value': 'Energy'},
                    ],
                    value='count',
                    style={'width': '200px', 'display': 'inline-block', 'backgroundColor': colors['background']}
                ),
            ], style={'marginBottom': '20px'}),
            
            dcc.Loading(
                id="loading-graph",
                type="circle",
                color=colors['accent'],
                children=[
                    dcc.Graph(
                        id='genre-network-graph', 
                        style={'height': '700px'},
                        config={'displayModeBar': True, 'scrollZoom': True}
                    ),
                ]
            ),
            
            html.Div([
                html.P("💡 Controls:", style={'color': colors['accent'], 'fontWeight': 'bold', 'marginBottom': '10px'}),
                html.P("• Click on nodes to select/deselect genres", style={'color': colors['text'], 'fontSize': '13px', 'margin': '5px 0'}),
                html.P("• Scroll to zoom in/out", style={'color': colors['text'], 'fontSize': '13px', 'margin': '5px 0'}),
                html.P("• Drag to rotate the 3D view", style={'color': colors['text'], 'fontSize': '13px', 'margin': '5px 0'}),
                html.P("• Hover over nodes to see genre details", style={'color': colors['text'], 'fontSize': '13px', 'margin': '5px 0'}),
            ], style={'backgroundColor': colors['background'], 'padding': '15px', 'borderRadius': '8px', 'marginTop': '15px'}),
        ])


#handle random genre generation
@app.callback(
    Output('selected-genres-store', 'data', allow_duplicate=True),
    Input('generate-random-btn', 'n_clicks'),
    State('random-genre-count', 'value'),
    prevent_initial_call=True
)
def generate_random_genres(n_clicks, count):
    if n_clicks > 0:
        return random.sample(all_genres, min(count, len(all_genres)))
    return []


#handle manual genre selection 
@app.callback(
    Output('selected-genres-store', 'data', allow_duplicate=True),
    [Input('manual-genre-selector', 'value'),
     Input('select-top-10', 'n_clicks'),
     Input('select-top-20', 'n_clicks'),
     Input('clear-selection', 'n_clicks')],
    prevent_initial_call=True
)
def update_manual_selection(selected, top10_clicks, top20_clicks, clear_clicks):
    #this callback is kept for backwards compatibility but won't be triggered
    return dash.no_update


#song search callback with ML predictions
@app.callback(
    Output('search-results', 'children'),
    Input('song-search', 'value'),
    prevent_initial_call=True
)
def search_songs(search_query):
    if not search_query or len(search_query) < 2:
        return html.P("Type at least 2 characters to search...", 
                     style={'color': colors['text'], 'fontStyle': 'italic'})

    search_lower = search_query.lower()
    artist_col = 'Artists' if 'Artists' in df.columns else ('Artist' if 'Artist' in df.columns else None)

    if artist_col:
        matches = df[
            df['Name'].str.lower().str.contains(search_lower, na=False) |
            df[artist_col].str.lower().str.contains(search_lower, na=False)
        ].head(10)
    else:
        matches = df[
            df['Name'].str.lower().str.contains(search_lower, na=False)
        ].head(10)

    if len(matches) == 0:
        return html.P(f"No songs found matching '{search_query}'", 
                     style={'color': colors['secondary'], 'fontStyle': 'italic'})

    results = []
    for idx, row in matches.iterrows():
        predictions_html = []

        if trained_models:
            predictions_html.append(
                html.Div([
                    html.H5(" ML Predictions:", 
                           style={'color': colors['accent'], 'marginTop': '15px', 'marginBottom': '10px', 'fontSize': '16px'})
                ])
            )

            for model_name, model_data in trained_models.items():
                try:
                    features_dict = {feat: row[feat] for feat in audio_features}
                    features_dict['Energy_x_Danceability'] = features_dict['Energy'] * features_dict['Danceability']
                    features_dict['Loudness_x_Energy'] = features_dict['Loudness'] * features_dict['Energy']
                    features_dict['Acousticness_x_Instrumentalness'] = features_dict['Acousticness'] * features_dict['Instrumentalness']
                    features_dict['Valence_x_Danceability'] = features_dict['Valence'] * features_dict['Danceability']
                    features_dict['Energy_squared'] = features_dict['Energy'] ** 2
                    features_dict['Tempo_normalized'] = features_dict['Tempo'] / 250.0

                    feature_order = model_data['feature_names']
                    features = np.array([features_dict.get(f, 0) for f in feature_order]).reshape(1, -1)

                    if model_data['type'] == 'hierarchical':
                        predicted_genre, confidence, top_3 = predict_with_hierarchical_model(features, model_data)
                    else:
                        model = model_data['model']
                        scaler = model_data['scaler']
                        label_encoder = model_data['label_encoder']
                        features_scaled = scaler.transform(features)
                        pred_idx = model.predict(features_scaled)[0]
                        predicted_genre = label_encoder.inverse_transform([pred_idx])[0]

                        if hasattr(model, 'predict_proba'):
                            probs = model.predict_proba(features_scaled)[0]
                            confidence = max(probs)
                            top_3_idx = np.argsort(probs)[-3:][::-1]
                            top_3_genres = label_encoder.inverse_transform(top_3_idx)
                            top_3_probs = probs[top_3_idx]
                            top_3 = list(zip(top_3_genres, top_3_probs))
                        else:
                            confidence = None
                            top_3 = [(predicted_genre, 1.0)]

                    is_correct = predicted_genre == row['Genre']

                    predictions_html.append(
                        html.Div([
                            html.Div([
                                html.Strong(f"{model_name}: ", style={'color': colors['text']}),
                                html.Span(
                                    predicted_genre,
                                    style={
                                        'color': colors['success'] if is_correct else colors['secondary'],
                                        'fontWeight': 'bold',
                                        'fontSize': '15px'
                                    }
                                ),
                                html.Span(
                                    f" ({confidence*100:.1f}%)" if confidence else "",
                                    style={'color': colors['text'], 'fontSize': '13px', 'marginLeft': '5px'}
                                ),
                                html.Span(
                                    " ✓" if is_correct else " ✗",
                                    style={
                                        'color': colors['success'] if is_correct else colors['secondary'],
                                        'fontWeight': 'bold',
                                        'fontSize': '16px',
                                        'marginLeft': '8px'
                                    }
                                ),
                            ], style={'marginBottom': '5px'}),
                            html.Div([
                                html.Span(f"  Top 3: ", style={'color': colors['text'], 'fontSize': '12px'}),
                                html.Span(
                                    " → ".join([f"{g} ({p*100:.0f}%)" for g, p in top_3]),
                                    style={'color': colors['text'], 'fontSize': '11px', 'opacity': '0.8'}
                                )
                            ], style={'marginBottom': '10px'})
                        ])
                    )
                except Exception as e:
                    predictions_html.append(
                        html.P(f"{model_name}: Prediction failed ({str(e)})",
                              style={'color': colors['text'], 'fontSize': '12px', 'opacity': '0.6'})
                    )
        else:
            predictions_html.append(
                html.P(" No models trained yet. Train a model to see predictions!",
                      style={'color': colors['secondary'], 'fontSize': '13px', 'fontStyle': 'italic', 'marginTop': '10px'})
            )

        results.append(
            html.Div([
                html.H4(f" {row['Name']}", 
                       style={'color': colors['accent'], 'margin': '0', 'fontSize': '18px'}),
                html.P(f"Artist: {row[artist_col]}" if artist_col else "", 
                      style={'color': colors['text'], 'margin': '5px 0', 'fontSize': '14px'}),
                html.P([
                    html.Span("Actual Genre: ", style={'color': colors['text']}),
                    html.Span(row['Genre'], style={'color': colors['success'], 'fontWeight': 'bold'})
                ], style={'margin': '5px 0', 'fontSize': '14px'}),
                html.Div([
                    html.Span(f"Energy: {row['Energy']:.2f}", 
                             style={'marginRight': '15px', 'color': colors['text'], 'fontSize': '12px'}),
                    html.Span(f"Danceability: {row['Danceability']:.2f}", 
                             style={'marginRight': '15px', 'color': colors['text'], 'fontSize': '12px'}),
                    html.Span(f"Tempo: {row['Tempo']:.0f}", 
                             style={'color': colors['text'], 'fontSize': '12px'}),
                ], style={'marginTop': '10px', 'paddingBottom': '10px', 'borderBottom': f'1px solid {colors["background"]}'}),

                html.Div(predictions_html)

            ], style={
                'backgroundColor': colors['card'],
                'padding': '15px',
                'borderRadius': '8px',
                'marginBottom': '15px',
                'border': f'2px solid {colors['accent']}'
            })
        )

    return html.Div([
        html.H4(f"Found {len(matches)} result(s):", 
               style={'color': colors['accent'], 'marginBottom': '15px'}),
        html.Div(results)
    ])



#generate genre network visualization 
@app.callback(
    Output('genre-network-graph', 'figure'),
    [Input('active-tab', 'data'),
     Input('color-by-feature', 'value'),
     Input('size-by-metric', 'value')],
    State('selected-genres-store', 'data')
)
def create_genre_network_3d(active_tab, color_feature, size_metric, selected_genres):
    if active_tab != 'visual':
        return go.Figure()

    print(" Rendering 3D visualization...")
    
    #use pre-computed data 
    G = GENRE_GRAPH
    pos = GENRE_POSITIONS
    genre_counts = GENRE_COUNTS
    
    #calculate color values from precomputed stats
    print(f"  Calculating colors by {color_feature}...")
    color_values = [GENRE_STATS[genre][color_feature] for genre in TOP_GENRES]
    
    #calculate sizes
    print(f"  Calculating sizes by {size_metric}...")
    if size_metric == 'count':
        node_sizes = [min(20, max(3, genre_counts[g] / 50)) for g in TOP_GENRES]
    else:
        node_sizes = [GENRE_STATS[g][size_metric] * 15 for g in TOP_GENRES]
    
    #extract 3D coordinates from precomputed positions
    print("  Creating node positions...")
    x_nodes = [pos[genre][0] for genre in TOP_GENRES]
    y_nodes = [pos[genre][1] for genre in TOP_GENRES]
    z_nodes = [pos[genre][2] for genre in TOP_GENRES]
    
    #node hover text
    node_text = [f"<b>{genre}</b><br>{genre_counts[genre]:,} songs<br>{color_feature}: {color_values[i]:.2f}" 
                 for i, genre in enumerate(TOP_GENRES)]
    
    #create edge traces
    print("  Creating edges...")
    edge_x, edge_y, edge_z = [], [], []
    edge_count = 0
    for edge in G.edges():
        if edge[0] in pos and edge[1] in pos:
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_z += [z0, z1, None]
            edge_count += 1
    
    print(f"  Created {edge_count} edges")

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color=colors['edge'], width=0.5),
        hoverinfo='none',
        showlegend=False
    )

    #determine node colors based on selection
    print("  Applying colors...")
    node_colors = []
    for i, genre in enumerate(TOP_GENRES):
        if selected_genres and genre in selected_genres:
            node_colors.append(colors['node_selected'])
        else:
            node_colors.append(color_values[i])
    
    #create node trace
    print("  Creating node trace...")
    node_trace = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=color_values if not selected_genres else node_colors,
            colorscale='Viridis' if not selected_genres else None,
            showscale=True if not selected_genres else False,
            colorbar=dict(
                title=color_feature,
                thickness=15,
                len=0.7,
                x=1.02
            ) if not selected_genres else None,
            line=dict(width=0.5, color='rgba(255,255,255,0.3)'),
            opacity=0.9
        ),
        text=[g for g in TOP_GENRES],
        hovertext=node_text,
        hoverinfo='text',
        customdata=TOP_GENRES,
        showlegend=False
    )

    print("  Building figure...")
    fig = go.Figure(data=[edge_trace, node_trace])
    
    fig.update_layout(
        title=dict(
            text=f"3D Genre Universe - {len(TOP_GENRES)} Genres<br><sub>Colored by {color_feature} | Sized by {size_metric}</sub>",
            font=dict(size=24, color=colors['accent'])
        ),
        margin=dict(l=0, r=0, b=0, t=60),
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['card'],
        font=dict(color=colors['text'], family='Arial, sans-serif'),
        scene=dict(
            xaxis=dict(
                showbackground=True,
                backgroundcolor='rgba(10, 14, 39, 0.5)',
                gridcolor='rgba(0, 217, 255, 0.1)',
                showticklabels=False,
                title=''
            ),
            yaxis=dict(
                showbackground=True,
                backgroundcolor='rgba(10, 14, 39, 0.5)',
                gridcolor='rgba(0, 217, 255, 0.1)',
                showticklabels=False,
                title=''
            ),
            zaxis=dict(
                showbackground=True,
                backgroundcolor='rgba(10, 14, 39, 0.5)',
                gridcolor='rgba(0, 217, 255, 0.1)',
                showticklabels=False,
                title=''
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        hovermode='closest',
        dragmode='orbit'
    )
    
    print(" Visualization ready!")
    return fig


#display selected genres
@app.callback(
    Output('selected-genres-display', 'children'),
    Input('selected-genres-store', 'data')
)
def display_selected_genres(selected):
    if not selected:
        return html.P("No genres selected yet.", style={'color': colors['text'], 'fontStyle': 'italic'})
    
    total_songs = len(df[df['Genre'].isin(selected)])
    
    return html.Div([
        html.P(f" {len(selected)} genres selected | {total_songs:,} total songs", 
               style={'color': colors['success'], 'fontWeight': 'bold', 'fontSize': '16px'}),
        html.Div([
            html.Span(f"{genre} ", 
                     style={'backgroundColor': colors['accent'], 'color': colors['background'],
                            'padding': '5px 12px', 'borderRadius': '15px', 'marginRight': '8px',
                            'marginBottom': '8px', 'display': 'inline-block', 'fontSize': '12px'})
            for genre in selected[:20]
        ]),
        html.P(f"...and {len(selected)-20} more" if len(selected) > 20 else "", 
               style={'color': colors['text'], 'fontSize': '12px', 'marginTop': '10px'})
    ])

#update selected genres from visual explorer
@app.callback(
    Output('selected-genres-store', 'data', allow_duplicate=True),
    Input('genre-network-graph', 'clickData'),
    State('selected-genres-store', 'data'),
    prevent_initial_call=True
)
def update_selected_from_click(click_data, current_selection):
    if not click_data or 'points' not in click_data:
        return dash.no_update
    
    try:
        #get the genre from the text field 
        clicked_genre = click_data['points'][0]['text']
        
        if not current_selection:
            current_selection = []
        
        #toggle selection
        if clicked_genre in current_selection:
            current_selection.remove(clicked_genre)
        else:
            current_selection.append(clicked_genre)
        
        return current_selection
    except (KeyError, IndexError) as e:
        print(f"Error selecting genre: {e}")
        return dash.no_update


#update model selector dropdown
@app.callback(
    Output('prediction-model-selector', 'options'),
    Output('prediction-model-selector', 'value'),
    Input('training-output', 'children')
)
def update_model_selector(training_output):
    if not trained_models:
        return [], None
    
    options = [{'label': name, 'value': name} for name in trained_models.keys()]
    default_value = list(trained_models.keys())[0] if trained_models else None
    return options, default_value


#train models callback
@app.callback(
    Output('training-output', 'children'),
    Input('train-button', 'n_clicks'),
    [State('model-selector', 'value'),
     State('selected-genres-store', 'data')],
    prevent_initial_call=True
)
def train_models(n_clicks, selected_models, selected_genres):
    if n_clicks == 0 or not selected_models:
        return ""
    
    if not selected_genres or len(selected_genres) == 0:
        return html.Div([
            html.P(" Please select genres first using the Genre Universe Explorer above!", 
                   style={'color': colors['secondary'], 'fontWeight': 'bold', 'fontSize': '16px'})
        ])
    
    df_train = df[df['Genre'].isin(selected_genres)].copy()
    
    if len(df_train) < 100:
        return html.Div([
            html.P(f" Not enough data! Selected genres only have {len(df_train)} songs. Please select more genres.", 
                   style={'color': colors['secondary'], 'fontWeight': 'bold'})
        ])
    
    #check class distribution
    genre_counts_train = df_train['Genre'].value_counts()
    min_samples = genre_counts_train.min()
    
    if min_samples < 5:
        return html.Div([
            html.P(f" Some genres have very few samples (min: {min_samples}). This may cause poor accuracy.", 
                   style={'color': colors['secondary'], 'fontWeight': 'bold'}),
            html.P(f"Genres with few samples: {', '.join(genre_counts_train[genre_counts_train < 10].index.tolist()[:10])}", 
                   style={'color': colors['text'], 'fontSize': '14px'})
        ])
    
    #feature engineering - create interaction features for better accuracy
    X = df_train[audio_features].copy()
    
    #add polynomial features
    X['Energy_x_Danceability'] = X['Energy'] * X['Danceability']
    X['Loudness_x_Energy'] = X['Loudness'] * X['Energy']
    X['Acousticness_x_Instrumentalness'] = X['Acousticness'] * X['Instrumentalness']
    X['Valence_x_Danceability'] = X['Valence'] * X['Danceability']
    X['Energy_squared'] = X['Energy'] ** 2
    X['Tempo_normalized'] = X['Tempo'] / 250.0  # Normalize tempo
    
    y = df_train['Genre']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    #use stratify only if we have enough samples per class
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
    except ValueError:
        #if stratify fails train without it
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models_config = {
        'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=25, random_state=42, n_jobs=-1, min_samples_split=5),
        'Logistic Regression': LogisticRegression(max_iter=3000, random_state=42, C=1.0, multi_class='ovr', solver='saga'),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True, C=1.0, gamma='auto'),
    }

    results = []

    for model_name in selected_models:
        if model_name in models_config:
            start_time = time.time()
            model = models_config[model_name]
            model.fit(X_train_scaled, y_train)

            accuracy = model.score(X_test_scaled, y_test)
            train_time = time.time() - start_time

            
            trained_models[model_name] = {
                'type': 'flat',  
                'model': model,
                'scaler': scaler,
                'label_encoder': label_encoder,
                'feature_names': X.columns.tolist()  #store feature names for prediction
            }

            results.append(
                html.Div([
                    html.H4(f" {model_name}", style={'color': colors['accent'], 'margin': '0'}),
                    html.P(f"Accuracy: {accuracy*100:.2f}%", style={'color': colors['text'], 'margin': '5px 0'}),
                    html.P(f"Training Time: {train_time:.2f}s", style={'color': colors['text'], 'margin': '5px 0'}),
                    html.P(f"Genres: {len(selected_genres)} | Songs: {len(df_train):,}", 
                           style={'color': colors['text'], 'margin': '5px 0', 'fontSize': '12px'}),
                    html.P(f"Min samples per genre: {min_samples}", 
                           style={'color': colors['text'], 'margin': '5px 0', 'fontSize': '12px'})
                ], style={'backgroundColor': colors['background'], 'padding': '15px', 
                         'borderRadius': '8px', 'marginBottom': '10px',
                         'border': f'1px solid {colors["accent"]}'})
            )

    return html.Div([
        html.H4("Training Complete!", style={'color': colors['accent']}),
        html.Div(results),
        html.Div([
            html.H4(" Understanding Your Accuracy:", style={'color': colors['accent'], 'marginTop': '25px'}),
            html.P(f"• With {len(selected_genres)} genres, random guessing would give ~{100/len(selected_genres):.1f}% accuracy", 
                   style={'color': colors['text'], 'fontSize': '14px', 'margin': '8px 0'}),
            html.P(f"• Your model achieved {accuracy*100:.1f}% - that's {accuracy/(1/len(selected_genres)):.1f}x better than random!", 
                   style={'color': colors['success'], 'fontSize': '14px', 'margin': '8px 0', 'fontWeight': 'bold'}),
            html.P(" Tips for better accuracy:", 
                   style={'color': colors['accent'], 'fontSize': '14px', 'marginTop': '15px', 'fontWeight': 'bold'}),
            html.P("  • Use 5-15 very distinct genres (e.g., rock, classical, hip hop, edm, jazz)", 
                   style={'color': colors['text'], 'fontSize': '13px', 'margin': '5px 0'}),
            html.P("  • Avoid similar subgenres (e.g., don't mix 'rock', 'indie rock', 'alternative rock')", 
                   style={'color': colors['text'], 'fontSize': '13px', 'margin': '5px 0'}),
            html.P("  • More genres = harder classification = lower accuracy (but still impressive!)", 
                   style={'color': colors['text'], 'fontSize': '13px', 'margin': '5px 0'}),
        ], style={'backgroundColor': colors['background'], 'padding': '20px', 'borderRadius': '10px', 'marginTop': '20px'})
    ])


#statistics cards
@app.callback(
    Output('stats-cards', 'children'),
    Input('genre-filter', 'value')
)
def update_stats_cards(selected_genres):
    if not selected_genres:
        return html.Div()
    
    df_filtered = df[df['Genre'].isin(selected_genres)]
    
    total_songs = len(df_filtered)
    avg_energy = df_filtered['Energy'].mean()
    avg_danceability = df_filtered['Danceability'].mean()
    
    cards = html.Div([
        html.Div([
            html.Div([
                html.H3(f"{total_songs:,}", style={'color': colors['accent'], 'margin': '0', 'fontSize': '36px'}),
                html.P("Total Songs", style={'color': colors['text'], 'margin': '5px 0'})
            ], style={'backgroundColor': colors['card'], 'padding': '20px', 'borderRadius': '10px', 
                     'textAlign': 'center', 'width': '30%', 'display': 'inline-block', 'marginRight': '3%'}),
            
            html.Div([
                html.H3(f"{avg_energy:.2f}", style={'color': colors['success'], 'margin': '0', 'fontSize': '36px'}),
                html.P("Avg Energy", style={'color': colors['text'], 'margin': '5px 0'})
            ], style={'backgroundColor': colors['card'], 'padding': '20px', 'borderRadius': '10px', 
                     'textAlign': 'center', 'width': '30%', 'display': 'inline-block', 'marginRight': '3%'}),
            
            html.Div([
                html.H3(f"{avg_danceability:.2f}", style={'color': colors['secondary'], 'margin': '0', 'fontSize': '36px'}),
                html.P("Avg Danceability", style={'color': colors['text'], 'margin': '5px 0'})
            ], style={'backgroundColor': colors['card'], 'padding': '20px', 'borderRadius': '10px', 
                     'textAlign': 'center', 'width': '30%', 'display': 'inline-block'}),
        ])
    ])
    
    return cards


#enre distribution
@app.callback(
    Output('genre-distribution', 'figure'),
    Input('genre-filter', 'value')
)
def update_genre_distribution(selected_genres):
    df_filtered = df[df['Genre'].isin(selected_genres)] if selected_genres else df
    
    #calculate average danceability per genre
    genre_danceability = df_filtered.groupby('Genre')['Danceability'].mean().sort_values(ascending=False).head(15)
    
    fig = px.bar(
        x=genre_danceability.index,
        y=genre_danceability.values,
        title="Top 15 Genres by Average Danceability",
        labels={'x': 'Genre', 'y': 'Average Danceability'}
    )
    
    fig.update_traces(marker_color=colors['accent'])
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['card'],
        font=dict(color=colors['text']),
        xaxis=dict(tickangle=-45)
    )
    
    return fig


#feature correlation
@app.callback(
    Output('feature-correlation', 'figure'),
    Input('genre-filter', 'value')
)
def update_correlation(selected_genres):
    df_filtered = df[df['Genre'].isin(selected_genres)] if selected_genres else df
    
    corr = df_filtered[audio_features].corr()
    
    fig = px.imshow(
        corr,
        title="Audio Feature Correlation Matrix",
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['card'],
        font=dict(color=colors['text'])
    )
    
    return fig


#scatter plot
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('genre-filter', 'value'),
     Input('feature-selector', 'value')]
)
def update_scatter(selected_genres, selected_features):
    df_filtered = df[df['Genre'].isin(selected_genres)] if selected_genres else df
    
    #make sure we have at least 2 features selected
    if not selected_features or len(selected_features) < 2:
        #default to first 2 audio features if none selected
        feat_x = audio_features[0]
        feat_y = audio_features[1]
    else:
        feat_x = selected_features[0]
        feat_y = selected_features[1]
    
    fig = px.scatter(
        df_filtered.sample(min(1000, len(df_filtered))),
        x=feat_x,
        y=feat_y,
        color='Genre',
        title=f"{feat_x} vs {feat_y}",
        opacity=0.6
    )
    
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['card'],
        font=dict(color=colors['text']),
        showlegend=True
    )
    
    return fig


#box plot
@app.callback(
    Output('box-plot', 'figure'),
    [Input('genre-filter', 'value'),
     Input('feature-selector', 'value')]
)
def update_boxplot(selected_genres, selected_features):
    df_filtered = df[df['Genre'].isin(selected_genres)] if selected_genres else df
    
    if not selected_features:
        return go.Figure()
    
    fig = px.box(
        df_filtered,
        x='Genre',
        y=selected_features[0] if selected_features else 'Energy',
        title=f"{selected_features[0] if selected_features else 'Energy'} Distribution by Genre"
    )
    
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['card'],
        font=dict(color=colors['text']),
        xaxis=dict(tickangle=-45)
    )
    
    return fig


#radar chart
@app.callback(
    Output('radar-chart', 'figure'),
    Input('genre-filter', 'value')
)
def update_radar(selected_genres):
    if not selected_genres:
        selected_genres = df['Genre'].value_counts().head(5).index.tolist()
    
    df_filtered = df[df['Genre'].isin(selected_genres[:5])]  #limit to 5 genres
    
    fig = go.Figure()
    
    for genre in selected_genres[:5]:
        genre_data = df[df['Genre'] == genre][audio_features].mean()
        
        fig.add_trace(go.Scatterpolar(
            r=genre_data.values,
            theta=audio_features,
            fill='toself',
            name=genre
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            bgcolor=colors['background']
        ),
        showlegend=True,
        title="Genre Audio Feature Profiles (Top 5 Selected)",
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['card'],
        font=dict(color=colors['text'])
    )
    
    return fig


#prediction callback
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [State('prediction-model-selector', 'value')] +
    [State(f'slider-{feat}', 'value') for feat in audio_features],
    prevent_initial_call=True
)
def predict_genre(n_clicks, model_name, *feature_values):
    if not model_name or model_name not in trained_models:
        return html.P(" Please train a model first!",
                      style={'color': colors['secondary'], 'fontWeight': 'bold'})

    model_data = trained_models[model_name]

    #map slider values to feature names
    features_dict = dict(zip(audio_features, feature_values))

    #add engineered features
    features_dict['Energy_x_Danceability'] = features_dict['Energy'] * features_dict['Danceability']
    features_dict['Loudness_x_Energy'] = features_dict['Loudness'] * features_dict['Energy']
    features_dict['Acousticness_x_Instrumentalness'] = features_dict['Acousticness'] * features_dict['Instrumentalness']
    features_dict['Valence_x_Danceability'] = features_dict['Valence'] * features_dict['Danceability']
    features_dict['Energy_squared'] = features_dict['Energy'] ** 2
    features_dict['Tempo_normalized'] = features_dict['Tempo'] / 250.0

    #build feature vector in correct order
    feature_order = model_data['feature_names']
    features = np.array([features_dict.get(f, 0) for f in feature_order]).reshape(1, -1)

    try:
        if model_data['type'] == 'hierarchical':
            predicted_genre, confidence, top_3 = predict_with_hierarchical_model(features, model_data)
            return html.Div([
                html.H4(f" Predicted Genre: {predicted_genre}"),
                html.P(f" Confidence: {confidence*100:.2f}%"),
                html.H5("Top 3 Predictions:"),
                html.Ul([html.Li(f"{g} ({p*100:.2f}%)") for g, p in top_3])
            ])
        else:
            model = model_data['model']
            scaler = model_data['scaler']
            label_encoder = model_data['label_encoder']

            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            predicted_genre = label_encoder.inverse_transform([prediction])[0]

            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(features_scaled)[0]
                confidence = max(probs)
                top_3_idx = np.argsort(probs)[-3:][::-1]
                top_3_genres = label_encoder.inverse_transform(top_3_idx)
                top_3_probs = probs[top_3_idx]

                return html.Div([
                    html.H4(f" Predicted Genre: {predicted_genre}"),
                    html.P(f" Confidence: {confidence*100:.2f}%"),
                    html.H5("Top 3 Predictions:"),
                    html.Ul([html.Li(f"{g} ({p*100:.2f}%)") for g, p in zip(top_3_genres, top_3_probs)])
                ])
            else:
                return html.H4(f" Predicted Genre: {predicted_genre}")

    except Exception as e:
        return html.Div(f" Prediction failed: {str(e)}")




# run the app
if __name__ == '__main__':
    app.run(debug=True, port=8050)