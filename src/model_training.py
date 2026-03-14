"""
Feature Engineering, Model Training, and Evaluation for Cricket Performance Prediction
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
from pathlib import Path
import json

# Create directories
Path("models").mkdir(exist_ok=True)
Path("preprocessors").mkdir(exist_ok=True)

print("="*60)
print("FEATURE ENGINEERING AND MODEL TRAINING")
print("="*60)

# Load cleaned data
print("\nLoading cleaned data...")
players_df = pd.read_csv("data/players_cleaned.csv")
matches_df = pd.read_csv("data/matches_cleaned.csv")

print(f"Players: {players_df.shape}")
print(f"Matches: {matches_df.shape}")

# =====================
# FEATURE ENGINEERING
# =====================
print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

# Merge player info with match performance
merged_df = matches_df.merge(players_df, on='player_id', how='left')

# Create new features
print("\nCreating new features...")

# Batting features
merged_df['strike_rate'] = np.where(
    merged_df['balls_faced'] > 0,
    (merged_df['runs_scored'] / merged_df['balls_faced']) * 100,
    0
)

merged_df['boundary_percentage'] = np.where(
    merged_df['balls_faced'] > 0,
    ((merged_df['fours'] + merged_df['sixes']) / merged_df['balls_faced']) * 100,
    0
)

merged_df['runs_per_boundary'] = np.where(
    (merged_df['fours'] + merged_df['sixes']) > 0,
    merged_df['runs_scored'] / (merged_df['fours'] + merged_df['sixes']),
    merged_df['runs_scored']
)

# Bowling features
merged_df['economy_rate'] = np.where(
    merged_df['overs_bowled'] > 0,
    merged_df['runs_conceded'] / merged_df['overs_bowled'],
    0
)

merged_df['wickets_per_over'] = np.where(
    merged_df['overs_bowled'] > 0,
    merged_df['wickets_taken'] / merged_df['overs_bowled'],
    0
)

merged_df['dot_ball_percentage'] = np.where(
    merged_df['overs_bowled'] > 0,
    (merged_df['dots'] / (merged_df['overs_bowled'] * 6)) * 100,
    0
)

# Fielding features
merged_df['total_fielding_contributions'] = (
    merged_df['catches'] + merged_df['run_outs'] + merged_df['stumpings']
)

# Performance score (composite metric)
merged_df['batting_performance_score'] = (
    merged_df['runs_scored'] * 1.0 +
    merged_df['fours'] * 2 +
    merged_df['sixes'] * 3 +
    merged_df['not_out'] * 10 +
    merged_df['strike_rate'] * 0.1
)

merged_df['bowling_performance_score'] = (
    merged_df['wickets_taken'] * 25 +
    merged_df['maidens'] * 10 +
    merged_df['dots'] * 2 -
    merged_df['runs_conceded'] * 0.5
)

merged_df['overall_performance_score'] = (
    merged_df['batting_performance_score'] + 
    merged_df['bowling_performance_score'] +
    merged_df['total_fielding_contributions'] * 5
)

# Experience features
merged_df['experience_level'] = pd.cut(
    merged_df['experience_years'],
    bins=[0, 2, 5, 10, 20],
    labels=['Rookie', 'Intermediate', 'Experienced', 'Veteran']
)

# Age groups
merged_df['age_group'] = pd.cut(
    merged_df['age'],
    bins=[0, 23, 27, 32, 50],
    labels=['Young', 'Prime', 'Mature', 'Senior']
)

# Form indicator (recent performance - simplified)
merged_df['is_home_venue'] = merged_df.apply(
    lambda x: 1 if x['country'] in ['India', 'Australia', 'England'] else 0, 
    axis=1
)

# Match importance (Test matches considered more important)
merged_df['match_importance'] = merged_df['match_type'].map({
    'Test': 3,
    'ODI': 2,
    'T20': 1
})

# Target variable: High performance (1) vs Normal/Low performance (0)
# Using overall_performance_score median as threshold
median_score = merged_df['overall_performance_score'].median()
merged_df['high_performance'] = (merged_df['overall_performance_score'] > median_score).astype(int)

print(f"Target variable distribution:\n{merged_df['high_performance'].value_counts()}")
print(f"Performance threshold (median score): {median_score:.2f}")

# =====================
# FEATURE SELECTION
# =====================
print("\n" + "="*60)
print("FEATURE SELECTION")
print("="*60)

# Select features for modeling
feature_columns = [
    # Player attributes
    'age', 'experience_years',
    
    # Batting features
    'balls_faced', 'runs_scored', 'fours', 'sixes', 'not_out',
    'strike_rate', 'boundary_percentage', 'runs_per_boundary',
    
    # Bowling features
    'overs_bowled', 'runs_conceded', 'wickets_taken', 'maidens', 'dots',
    'economy_rate', 'wickets_per_over', 'dot_ball_percentage',
    
    # Fielding features
    'catches', 'run_outs', 'stumpings', 'total_fielding_contributions',
    
    # Performance scores
    'batting_performance_score', 'bowling_performance_score',
    
    # Context features
    'match_importance', 'is_home_venue'
]

# Encode categorical variables
print("\nEncoding categorical variables...")
le_role = LabelEncoder()
le_batting_style = LabelEncoder()
le_bowling_style = LabelEncoder()
le_match_type = LabelEncoder()
le_venue = LabelEncoder()
le_experience_level = LabelEncoder()
le_age_group = LabelEncoder()

merged_df['role_encoded'] = le_role.fit_transform(merged_df['role'].fillna('Unknown'))
merged_df['batting_style_encoded'] = le_batting_style.fit_transform(merged_df['batting_style'].fillna('Unknown'))
merged_df['bowling_style_encoded'] = le_bowling_style.fit_transform(merged_df['bowling_style'].fillna('Unknown'))
merged_df['match_type_encoded'] = le_match_type.fit_transform(merged_df['match_type'])
merged_df['venue_encoded'] = le_venue.fit_transform(merged_df['venue'])
merged_df['experience_level_encoded'] = le_experience_level.fit_transform(merged_df['experience_level'].astype(str))
merged_df['age_group_encoded'] = le_age_group.fit_transform(merged_df['age_group'].astype(str))

# Add encoded features
feature_columns.extend([
    'role_encoded', 'batting_style_encoded', 'bowling_style_encoded',
    'match_type_encoded', 'venue_encoded', 'experience_level_encoded', 'age_group_encoded'
])

# Prepare feature matrix
X = merged_df[feature_columns].copy()
y = merged_df['high_performance'].copy()

# Handle any remaining NaN values
X = X.fillna(0)

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target variable shape: {y.shape}")
print(f"Number of features: {len(feature_columns)}")

# =====================
# TRAIN-TEST SPLIT
# =====================
print("\n" + "="*60)
print("TRAIN-TEST SPLIT")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Train/Test split: {X_train.shape[0]/len(X)*100:.1f}%/{X_test.shape[0]/len(X)*100:.1f}%")

# Feature scaling
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================
# MODEL TRAINING
# =====================
print("\n" + "="*60)
print("MODEL TRAINING")
print("="*60)

# Define models
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.1,
        random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0
    )
}

# Train and evaluate models
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'auc_roc': auc_roc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC-ROC: {auc_roc:.4f}")

# =====================
# BEST MODEL SELECTION
# =====================
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

# Find best model based on accuracy
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']

print(f"\nBest Model: {best_model_name}")
print(f"Best Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"Best AUC-ROC: {results[best_model_name]['auc_roc']:.4f}")

# =====================
# DETAILED EVALUATION
# =====================
print("\n" + "="*60)
print("DETAILED EVALUATION")
print("="*60)

# Use best model for detailed evaluation
if best_model_name == 'Logistic Regression':
    y_pred_best = best_model.predict(X_test_scaled)
else:
    y_pred_best = best_model.predict(X_test)

print(f"\nClassification Report ({best_model_name}):")
print(classification_report(y_test, y_pred_best))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(cm)

# Feature importance (for tree-based models)
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE")
    print("="*60)
    
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15).to_string(index=False))
    
    # Save feature importance
    feature_importance.to_csv("models/feature_importance.csv", index=False)
    print("\n✓ Saved: models/feature_importance.csv")

# =====================
# SAVE MODELS AND PREPROCESSORS
# =====================
print("\n" + "="*60)
print("SAVING MODELS AND PREPROCESSORS")
print("="*60)

# Save best model
joblib.dump(best_model, f"models/best_model_{best_model_name.replace(' ', '_')}.joblib")
print(f"✓ Saved: models/best_model_{best_model_name.replace(' ', '_')}.joblib")

# Save all models for comparison
for name, data in results.items():
    joblib.dump(data['model'], f"models/{name.replace(' ', '_')}.joblib")
    print(f"✓ Saved: models/{name.replace(' ', '_')}.joblib")

# Save scaler
joblib.dump(scaler, "models/scaler.joblib")
print("✓ Saved: models/scaler.joblib")

# Save label encoders
encoders = {
    'role': le_role,
    'batting_style': le_batting_style,
    'bowling_style': le_bowling_style,
    'match_type': le_match_type,
    'venue': le_venue,
    'experience_level': le_experience_level,
    'age_group': le_age_group
}

joblib.dump(encoders, "models/label_encoders.joblib")
print("✓ Saved: models/label_encoders.joblib")

# Save feature columns
with open("models/feature_columns.json", "w") as f:
    json.dump(feature_columns, f)
print("✓ Saved: models/feature_columns.json")

# Save model metadata
metadata = {
    "best_model": best_model_name,
    "accuracy": float(results[best_model_name]['accuracy']),
    "auc_roc": float(results[best_model_name]['auc_roc']),
    "feature_count": len(feature_columns),
    "train_samples": int(X_train.shape[0]),
    "test_samples": int(X_test.shape[0]),
    "threshold": float(median_score)
}

with open("models/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("✓ Saved: models/model_metadata.json")

# Save evaluation results
eval_results = {
    "models": {name: {"accuracy": float(data['accuracy']), "auc_roc": float(data['auc_roc'])} 
               for name, data in results.items()},
    "classification_report": classification_report(y_test, y_pred_best, output_dict=True),
    "confusion_matrix": cm.tolist()
}

with open("models/evaluation_results.json", "w") as f:
    json.dump(eval_results, f, indent=2)
print("✓ Saved: models/evaluation_results.json")

print("\n" + "="*60)
print("MODEL TRAINING COMPLETE!")
print("="*60)
print(f"\nBest Model: {best_model_name}")
print(f"Test Accuracy: {results[best_model_name]['accuracy']*100:.2f}%")
print(f"AUC-ROC Score: {results[best_model_name]['auc_roc']*100:.2f}%")
print("\nAll models and preprocessors saved to: models/")
