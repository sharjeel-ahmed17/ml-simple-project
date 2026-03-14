"""
FastAPI Application for Cricket Performance Prediction
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Initialize FastAPI app
app = FastAPI(
    title="Cricket Performance Prediction API",
    description="Predict cricket player performance using ML models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and preprocessors
print("Loading models and preprocessors...")

MODEL_PATH = Path("models")

# Load best model
try:
    model = joblib.load(MODEL_PATH / "best_model_Gradient_Boosting.joblib")
    scaler = joblib.load(MODEL_PATH / "scaler.joblib")
    encoders = joblib.load(MODEL_PATH / "label_encoders.joblib")
    
    with open(MODEL_PATH / "feature_columns.json", "r") as f:
        feature_columns = json.load(f)
    
    with open(MODEL_PATH / "model_metadata.json", "r") as f:
        metadata = json.load(f)
    
    print("✓ Models loaded successfully!")
except Exception as e:
    print(f"✗ Error loading models: {e}")
    raise

# Pydantic models for request/response
class PlayerMatchInput(BaseModel):
    # Player attributes
    age: int = Field(..., ge=18, le=50, description="Player age")
    experience_years: int = Field(..., ge=0, le=30, description="Years of experience")
    role: str = Field(..., description="Player role: Batsman, Bowler, All-rounder, Wicket-keeper")
    batting_style: str = Field(..., description="Batting style: Right-hand bat, Left-hand bat")
    bowling_style: str = Field(..., description="Bowling style")
    
    # Match context
    match_type: str = Field(..., description="Match type: T20, ODI, Test")
    venue: str = Field(..., description="Match venue")
    opposition: str = Field(..., description="Opposition country")
    
    # Batting performance
    balls_faced: int = Field(default=0, ge=0, description="Balls faced")
    runs_scored: int = Field(default=0, ge=0, description="Runs scored")
    fours: int = Field(default=0, ge=0, description="Number of fours")
    sixes: int = Field(default=0, ge=0, description="Number of sixes")
    not_out: int = Field(default=0, ge=0, le=1, description="Not out (1) or out (0)")
    
    # Bowling performance
    overs_bowled: float = Field(default=0.0, ge=0, description="Overs bowled")
    runs_conceded: int = Field(default=0, ge=0, description="Runs conceded")
    wickets_taken: int = Field(default=0, ge=0, le=10, description="Wickets taken")
    maidens: int = Field(default=0, ge=0, description="Maiden overs")
    dots: int = Field(default=0, ge=0, description="Dot balls")
    
    # Fielding performance
    catches: int = Field(default=0, ge=0, description="Catches taken")
    run_outs: int = Field(default=0, ge=0, le=5, description="Run outs")
    stumpings: int = Field(default=0, ge=0, le=5, description="Stumpings")


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Performance prediction: High or Normal/Low")
    probability: float = Field(..., description="Probability of high performance")
    confidence: str = Field(..., description="Confidence level")
    performance_score_threshold: float = Field(..., description="Threshold used for prediction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "High Performance",
                "probability": 0.85,
                "confidence": "High",
                "performance_score_threshold": 150.24
            }
        }


class ModelInfo(BaseModel):
    model_name: str
    accuracy: float
    auc_roc: float
    feature_count: int
    version: str


def calculate_derived_features(data: dict) -> dict:
    """Calculate derived features from raw input"""
    
    # Batting features
    data['strike_rate'] = (data['runs_scored'] / data['balls_faced'] * 100) if data['balls_faced'] > 0 else 0
    data['boundary_percentage'] = ((data['fours'] + data['sixes']) / data['balls_faced'] * 100) if data['balls_faced'] > 0 else 0
    data['runs_per_boundary'] = (data['runs_scored'] / (data['fours'] + data['sixes'])) if (data['fours'] + data['sixes']) > 0 else data['runs_scored']
    
    # Bowling features
    data['economy_rate'] = (data['runs_conceded'] / data['overs_bowled']) if data['overs_bowled'] > 0 else 0
    data['wickets_per_over'] = (data['wickets_taken'] / data['overs_bowled']) if data['overs_bowled'] > 0 else 0
    data['dot_ball_percentage'] = ((data['dots'] / (data['overs_bowled'] * 6)) * 100) if data['overs_bowled'] > 0 else 0
    
    # Fielding features
    data['total_fielding_contributions'] = data['catches'] + data['run_outs'] + data['stumpings']
    
    # Performance scores
    data['batting_performance_score'] = (
        data['runs_scored'] * 1.0 +
        data['fours'] * 2 +
        data['sixes'] * 3 +
        data['not_out'] * 10 +
        data['strike_rate'] * 0.1
    )
    
    data['bowling_performance_score'] = (
        data['wickets_taken'] * 25 +
        data['maidens'] * 10 +
        data['dots'] * 2 -
        data['runs_conceded'] * 0.5
    )
    
    # Context features
    data['match_importance'] = {'Test': 3, 'ODI': 2, 'T20': 1}.get(data['match_type'], 1)
    data['is_home_venue'] = 1 if data.get('country', '') in ['India', 'Australia', 'England'] else 0
    
    return data


def encode_features(data: dict) -> dict:
    """Encode categorical features"""
    try:
        data['role_encoded'] = encoders['role'].transform([data['role']])[0]
    except:
        data['role_encoded'] = 0
    
    try:
        data['batting_style_encoded'] = encoders['batting_style'].transform([data['batting_style']])[0]
    except:
        data['batting_style_encoded'] = 0
    
    try:
        data['bowling_style_encoded'] = encoders['bowling_style'].transform([data['bowling_style']])[0]
    except:
        data['bowling_style_encoded'] = 0
    
    try:
        data['match_type_encoded'] = encoders['match_type'].transform([data['match_type']])[0]
    except:
        data['match_type_encoded'] = 0
    
    try:
        data['venue_encoded'] = encoders['venue'].transform([data['venue']])[0]
    except:
        data['venue_encoded'] = 0
    
    # Default encoding for derived categorical features
    data['experience_level_encoded'] = 0
    data['age_group_encoded'] = 0
    
    return data


@app.get("/", tags=["Home"])
async def home():
    """API Home endpoint"""
    return {
        "message": "Welcome to Cricket Performance Prediction API",
        "docs": "/docs",
        "version": "1.0.0"
    }


@app.get("/api/model-info", response_model=ModelInfo, tags=["Model Info"])
async def get_model_info():
    """Get information about the loaded model"""
    return ModelInfo(
        model_name=metadata.get("best_model", "Unknown"),
        accuracy=metadata.get("accuracy", 0),
        auc_roc=metadata.get("auc_roc", 0),
        feature_count=metadata.get("feature_count", 0),
        version="1.0.0"
    )


@app.post("/api/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_performance(input_data: PlayerMatchInput):
    """
    Predict player performance based on match statistics
    
    - **age**: Player's age
    - **experience_years**: Years of playing experience
    - **role**: Player role (Batsman, Bowler, All-rounder, Wicket-keeper)
    - **match_type**: Type of match (T20, ODI, Test)
    - **runs_scored**: Runs scored in the match
    - **wickets_taken**: Wickets taken in the match
    - And more...
    """
    try:
        # Convert to dictionary
        data = input_data.dict()
        
        # Calculate derived features
        data = calculate_derived_features(data)
        
        # Encode categorical features
        data = encode_features(data)
        
        # Create feature vector
        feature_vector = [data.get(col, 0) for col in feature_columns]
        feature_array = np.array(feature_vector).reshape(1, -1)
        
        # Scale features
        feature_scaled = scaler.transform(feature_array)
        
        # Make prediction
        prediction = model.predict(feature_scaled)[0]
        probability = model.predict_proba(feature_scaled)[0][1]
        
        # Determine confidence level
        if probability >= 0.8:
            confidence = "High"
        elif probability >= 0.6:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Format prediction
        prediction_label = "High Performance" if prediction == 1 else "Normal/Low Performance"
        
        return PredictionResponse(
            prediction=prediction_label,
            probability=float(round(probability, 4)),
            confidence=confidence,
            performance_score_threshold=metadata.get("threshold", 150.24)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/api/batch-predict", tags=["Prediction"])
async def batch_predict(input_data: List[PlayerMatchInput]):
    """
    Batch prediction for multiple player performances
    """
    try:
        results = []
        
        for item in input_data:
            data = item.dict()
            data = calculate_derived_features(data)
            data = encode_features(data)
            
            feature_vector = [data.get(col, 0) for col in feature_columns]
            feature_array = np.array(feature_vector).reshape(1, -1)
            feature_scaled = scaler.transform(feature_array)
            
            prediction = model.predict(feature_scaled)[0]
            probability = model.predict_proba(feature_scaled)[0][1]
            
            results.append({
                "prediction": "High Performance" if prediction == 1 else "Normal/Low Performance",
                "probability": float(round(probability, 4))
            })
        
        return {"predictions": results, "count": len(results)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/api/features", tags=["Model Info"])
async def get_feature_list():
    """Get list of features used by the model"""
    return {
        "features": feature_columns,
        "count": len(feature_columns)
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
