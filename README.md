# 🏏 Cricket Performance Predictor

A complete machine learning project for predicting cricket player performance using FastAPI and Streamlit.

## 📋 Project Structure

```
ml-simple-project/
├── data/                    # Data files
│   ├── players.csv         # Player information
│   ├── matches.csv         # Match performance data
│   └── *_cleaned.csv       # Cleaned data files
├── models/                  # Trained models and preprocessors
│   ├── best_model_*.joblib # Best performing model
│   ├── scaler.joblib       # Feature scaler
│   └── *.json              # Model metadata
├── src/                     # Source code
│   ├── data_generation.py  # Generate synthetic data
│   ├── eda.py              # EDA and visualizations
│   └── model_training.py   # Feature engineering & training
├── app/                     # Application files
│   ├── main.py             # FastAPI application
│   └── streamlit_app.py    # Streamlit UI
├── visualizations/          # EDA charts
├── eda_reports/             # EDA reports
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Data

```bash
python src/data_generation.py
```

### 3. Run EDA

```bash
python src/eda.py
```

### 4. Train Model

```bash
python src/model_training.py
```

### 5. Start FastAPI Server

```bash
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API Docs: http://localhost:8000/docs

### 6. Start Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

## 📊 Features

### Data Generation
- Synthetic cricket player data (100 players)
- Match performance records (11,000+ matches)
- Realistic statistics for batting, bowling, and fielding

### EDA & Visualization
- Player demographics analysis
- Performance distributions
- Correlation analysis
- 12+ interactive visualizations

### Machine Learning
- **Feature Engineering**: 33 features including:
  - Batting metrics (strike rate, boundary %)
  - Bowling metrics (economy, wickets/over)
  - Performance scores
  - Context features
- **Models Trained**:
  - Random Forest
  - Gradient Boosting ⭐ (Best: 99.6% accuracy)
  - Logistic Regression
- **Evaluation**: Accuracy, AUC-ROC, Classification Report

### FastAPI Backend
- RESTful API for predictions
- Single and batch prediction endpoints
- Model information endpoint
- Auto-generated API docs

### Streamlit UI
- Interactive prediction form
- Analytics dashboard
- Real-time visualizations
- Sample scenarios

## 🎯 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message |
| `/api/predict` | POST | Single prediction |
| `/api/batch-predict` | POST | Batch predictions |
| `/api/model-info` | GET | Model information |
| `/api/features` | GET | Feature list |
| `/health` | GET | Health check |

## 📈 Model Performance

| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| Gradient Boosting | 99.64% | 99.69% |
| Logistic Regression | 99.64% | 99.99% |
| Random Forest | 99.59% | 99.99% |

## 🔮 Prediction Input

```json
{
  "age": 28,
  "experience_years": 5,
  "role": "All-rounder",
  "batting_style": "Right-hand bat",
  "bowling_style": "Right-arm fast",
  "match_type": "T20",
  "venue": "Mumbai",
  "opposition": "Australia",
  "balls_faced": 50,
  "runs_scored": 60,
  "fours": 6,
  "sixes": 2,
  "not_out": 1,
  "overs_bowled": 4.0,
  "runs_conceded": 30,
  "wickets_taken": 2,
  "maidens": 0,
  "dots": 12,
  "catches": 1,
  "run_outs": 0,
  "stumpings": 0
}
```

## 🛠️ Technologies

- **Python 3.8+**
- **Data**: Pandas, NumPy
- **ML**: Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit

## 📝 License

MIT License

---

**Created with ❤️ for Cricket Analytics**
