# Music Genre Classification Dashboard

An interactive dashboard that uses  machine learning to classify music genres based on audio features. Built with Python, Dash, and scikit-learn, this project implements flat and hierarchical classification models to predict genres across more than 5000 music genres.

![Dashboard Preview](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## Features

- **Interactive 3D Genre Explorer**: Visualize 5000+ genres in a 3D space 
- **Hierarchical Classification**: Two-level classification system (parent genres → subgenres) for improved accuracy
- **Multiple ML Models**: Random Forest, Logistic Regression, and SVM classifiers
- **Real-time Predictions**: Search for songs and get instant genre predictions with confidence scores
- **Manual Genre Prediction**: Adjust audio feature sliders to predict genres from custom inputs
- **Visualizations**: 
  - Genre distribution charts
  - Feature correlation matrices
  - Scatter plots and box plots
  - ROC curves and feature importance analysis

### The Challenge

This project is set to adress the problem of: **distinguishing between more than 5,000 specific genres**.  

The hierarchical model achieves **10.09% accuracy across all 5,000+ genres** ( **303 times better than random guessing**). While this might seem low, we should consider it to be remarkable given that:
- Random guessing would achieve only 0.03% accuracy
- Many genres are extremely similar (e.g., 411 types of indie, 355 types of hip hop)
- Audio features alone can't capture all the nuances humans use to classify music

For practical applications with 5-20 carefully selected, distinct genres, the models achieve 50-85% accuracy.

## Dataset

This project uses a comprehensive music dataset with:
- **375,000+** songs
- **5,000+** genres
- **Audio features**: Danceability, Energy, Loudness, Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Tempo

### Download the Dataset

The dataset is not included in this repository due to size constraints. Below are ways to obtain it:

**Option 1**: Extract from the included zip file:
```bash
cd data
unzip songs.zip
```

**Option 2**: Download from [source](https://www.kaggle.com/datasets/nikitricky/every-noise-at-once?resource=download) (if you need a fresh copy):
- Place the `songs.csv` file in the `data/` directory
- The file should contain columns: `Name`, `Artists`, `Genre`, `Release`, and audio features

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended (8GB+ for training on full dataset)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/0any13/music-genre-classifier.git
cd music-genre-classifier
```

2. **Create a virtual environment** 
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install required packages**
```bash
pip install pandas numpy matplotlib seaborn plotly dash scikit-learn networkx psutil
```


4. **Extract the dataset**
```bash
cd data
unzip songs.zip
cd ..
```

## Training Models

The pre-trained models are not included in this repository. You'll need to train them yourself using the provided notebook.

### Quick Training (Flat Model)

1. Open `notebooks/01_exploration.ipynb`
2. Run cells up to **"MAXIMUM GENRE COVERAGE STRATEGY"** section
3. This will train a flat Random Forest classifier and save it to `models/`

### Advanced Training (Hierarchical Model)

1. Run **"Smart Genre Hierarchy Builder"** (~45 seconds)
2. Run **"TRAIN LEVEL 1"** - Parent genre classifier (~35 seconds)
3. Run **"TRAIN LEVEL 2"** - Subgenre classifiers (~3 minutes)(depending on number of parents)
4. Run **"TEST COMPLETE HIERARCHICAL SYSTEM"** - Optional evaluation (~80 minutes)

**Performance:**
- The hierarchical model achieves **10.09% overall accuracy** across 5,000+ subgenres
- Level 1 (parent genre) accuracy: **15.79%**
- This is **303.5x better** than random guessing (0.03% baseline)
- Individual Level 2 classifiers achieve 9-42% accuracy within their parent categories
- Performance varies significantly based on genre complexity and number of subgenres

**Level 1 (Parent Genre) Individual Accuracies:**
- Latin: 38.1%
- Country: 22.0%
- Reggae: 17.8%
- Hip Hop: 16.8%
- Classical: 16.5%
- Indie: 16.4%
- Pop: 15.8%
- Jazz: 15.6%
- Rock: 14.9%
- Electro: 13.5%
- Folk: 12.9%
- Metal: 12.2%
- Trap: 11.5%
- Punk: 9.4%
...

**Level 2 (Subgenre) Performance by Parent** (sorted by accuracy):
- Emo: 42.4% across 30 subgenres
- Latin: 40.6% across 19 subgenres
- Alternative: 39.3% across 30 subgenres
- Ska: 36.5% across 32 subgenres
- Country: 34.6% across 40 subgenres
- Black Metal: 32.3% across 63 subgenres
- Hardcore: 30.5% across 46 subgenres
- Reggae: 28.8% across 45 subgenres
- Techno: 26.9% across 55 subgenres
- Trap: 26.1% across 64 subgenres
- Jazz: 22.8% across 134 subgenres
- Electro: 22.0% across 106 subgenres
- House: 20.9% across 89 subgenres
- Classical: 20.4% across 157 subgenres
- Folk: 18.1% across 164 subgenres
- Metal: 17.2% across 204 subgenres
- Punk: 17.0% across 169 subgenres
...

## Running the Dashboard

After training your models:

```bash
cd dashboard
python app.py
```

Then open your browser to: **http://127.0.0.1:8050**

### Dashboard Features

1. **Genre Universe Explorer** - 3D visualization of 5000+ genres, select via clicking or random sampling
2. **Model Training** - Train RF/LR/SVM on custom genre selections, view accuracy and timing
3. **Visualizations** - Genre distributions, feature correlations, scatter plots, radar charts
4. **Song Search** - Look up songs to see actual vs. predicted genres with confidence scores
5. **Manual Prediction** - Adjust feature sliders to predict genres from custom audio characteristics Experiment with different feature values

## Project Structure

```
music-genre-classifier/
│
├── dashboard/
│   └── app.py                          # Main Dash application
│
├── data/
│   ├── songs.csv                       # Main dataset (extract from songs.zip)
│   ├── songs.zip                       # Compressed dataset (51 MB)
│   ├── genre_hierarchy.csv             # Genre parent-child mappings
│   ├── recommended_parents.txt         # Suggested parent genres
│   └── all_genres_list.txt             # Complete genre list (5000+)
│
├── models/                             # Generated during training (not in repo)
│   ├── genre_classifier.pkl            # Flat model
│   ├── scaler.pkl                      # Feature scaler
│   ├── label_encoder.pkl               # Label encoder
│   │
│   └── hierarchical/                   # Hierarchical model components
│       ├── parent_model.pkl            # Level 1: Parent classifier
│       ├── parent_scaler.pkl           # Parent scaler
│       ├── parent_encoder.pkl          # Parent encoder
│       ├── metadata.pkl                # Model metadata
│       │
│       └── [21 genre-specific models]  # Level 2: Specialized classifiers
│           # Each parent genre has 3 files: _model.pkl, _scaler.pkl, _encoder.pkl
│           # Examples: alternative_*, black_metal_*, classical_*, country_*,
│           # electro_*, emo_*, folk_*, hardcore_*, hip_hop_*, house_*, indie_*,
│           # jazz_*, latin_*, metal_*, pop_*, punk_*, reggae_*, rock_*,
│           # ska_*, techno_*, trap_*
│
├── notebooks/
│   ├── 01_exploration.ipynb            # Full EDA and model training
│   ├── correlation_matrix.pdf          # Feature correlations
│   └── geographical.png                # Geographic distribution map
│
├── .gitignore
└── README.md
```

**Note:** All `.pkl` model files are generated during training and are not included in the repository due to size constraints.

## Model Architecture

### Flat Classification
- **Algorithm**: Random Forest (300 trees, max depth 25)
- **Features**: 9 base audio features + 6 engineered features
- **Use Case**: Quick training, good for < 20 distinct genres

### Hierarchical Classification
- **Level 1**: Classify into parent genres (rock, pop, hip hop, etc.)
- **Level 2**: Specialized classifiers for subgenres within each parent
- **Advantages**: 
  - Better accuracy on large genre sets (20+ genres)
  - Mimics human music categorization
  - More interpretable predictions

### Audio Features Used

**Base Features** :
- **Danceability**: How suitable a track is for dancing (0.0 to 1.0)
- **Energy**: Intensity and activity measure (0.0 to 1.0)
- **Loudness**: Overall loudness in decibels (-60 to 0 dB)
- **Speechiness**: Presence of spoken words (0.0 to 1.0)
- **Acousticness**: Confidence the track is acoustic (0.0 to 1.0)
- **Instrumentalness**: Predicts if track contains no vocals (0.0 to 1.0)
- **Liveness**: Presence of audience (0.0 to 1.0)
- **Valence**: Musical positiveness (0.0 to 1.0)
- **Tempo**: Estimated tempo in BPM (0 to 250)



### Expected Accuracy Benchmarks:

**Important Note:** This project approaches a classification problem with 5,000+ genres. The results below reflect real-world performance:

| Scenario | Number of Genres | Accuracy | vs Random Baseline |
|----------|-----------------|----------|-------------------|
| **Full Dataset (Hierarchical)** | 5,000+ subgenres | 10.09% | **303x better** |
| **Level 1 Only** | 21 parent genres | 15.79% | **3.3x better** |
| **Level 2 Individual Performance** | Varies by parent | 18-42% | **Varies** |

**Level 2 Performance Examples** :
- **Emo**: 42.4% accuracy across 30 subgenres
- **Latin**: 40.6% across 19 subgenres
- **Alternative**: 39.3% across 30 subgenres  
- **Ska**: 36.5% across 32 subgenres
- **Country**: 34.6% across 40 subgenres
- **Black Metal**: 32.3% across 63 subgenres
- **Hardcore**: 30.5% across 46 subgenres
- **Reggae**: 28.8% across 45 subgenres

**Challenging categories** (many similar subgenres):
- **Pop**: 9.2% across 417 subgenres
- **Indie**: 11.2% across 411 subgenres
- **Rock**: 10.5% across 379 subgenres
- **Hip Hop**: 10.1% across 355 subgenres


**For Smaller, Practical Use Cases** :

| Number of Genres | Random Baseline | Expected Accuracy |
|-----------------|-----------------|-------------------|
| 5 distinct genres | 20% | 75-85% |
| 10 distinct genres | 10% | 55-70% |
| 20 distinct genres | 5% | 35-50% |

**Example: Dashboard Training Results** (12 genres, 1,200 songs, ~100 songs per genre):

| Model | Accuracy | Training Time | vs Random (8.3%) |
|-------|----------|---------------|------------------|
| Random Forest | 62.5% | 0.73s | 7.5x better |
| SVM | 60.0% | 0.15s | 7.2x better |
| Logistic Regression | 57.5% | 0.24s | 6.9x better |
.

## Exploratory Analysis Highlights

The `01_exploration.ipynb` notebook includes:

- **Geographic Analysis**: Genres by country (using keyword extraction)
- **Temporal Trends**: Genre evolution from 1960s to 2020s
- **Feature Correlations**: How audio features relate to each other
- **Feature Importance**: Which features matter most for classification

