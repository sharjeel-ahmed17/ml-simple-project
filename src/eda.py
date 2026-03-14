"""
Exploratory Data Analysis (EDA) and Data Cleaning for Cricket Performance Data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Create directories for outputs
Path("eda_reports").mkdir(exist_ok=True)
Path("visualizations").mkdir(exist_ok=True)

# Load data
print("Loading data...")
players_df = pd.read_csv("data/players.csv")
matches_df = pd.read_csv("data/matches.csv")

print(f"Players shape: {players_df.shape}")
print(f"Matches shape: {matches_df.shape}")

# =====================
# DATA CLEANING
# =====================
print("\n" + "="*50)
print("DATA CLEANING")
print("="*50)

# Check for missing values
print("\nMissing Values in Players Data:")
print(players_df.isnull().sum())

print("\nMissing Values in Matches Data:")
print(matches_df.isnull().sum())

# Handle missing values
# For dismissal column, fill NaN with 'not out'
matches_df['dismissal'] = matches_df['dismissal'].fillna('not out')

# Check for duplicates
print(f"\nDuplicate rows in matches: {matches_df.duplicated().sum()}")

# Remove duplicates if any
matches_df = matches_df.drop_duplicates()

# Data type conversions
matches_df['match_date'] = pd.to_datetime(matches_df['match_date'])
players_df['age'] = players_df['age'].astype(int)
players_df['experience_years'] = players_df['experience_years'].astype(int)

# Check for outliers using IQR method
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Detect outliers in key columns
outlier_cols = ['runs_scored', 'balls_faced', 'overs_bowled', 'runs_conceded']
print("\nOutlier Detection:")
for col in outlier_cols:
    outliers, lower, upper = detect_outliers_iqr(matches_df, col)
    print(f"{col}: {len(outliers)} outliers (bounds: {lower:.2f}, {upper:.2f})")

# Cap extreme outliers (optional - keeping data for now)
# matches_df['runs_scored'] = matches_df['runs_scored'].clip(lower=0, upper=200)

# =====================
# EXPLORATORY DATA ANALYSIS
# =====================
print("\n" + "="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)

# Basic statistics
print("\nPlayers Data Statistics:")
print(players_df.describe())

print("\nMatches Data Statistics:")
print(matches_df.describe())

# Player distribution by country
print("\nPlayers by Country:")
print(players_df['country'].value_counts())

# Player distribution by role
print("\nPlayers by Role:")
print(players_df['role'].value_counts())

# Match type distribution
print("\nMatches by Type:")
print(matches_df['match_type'].value_counts())

# =====================
# VISUALIZATIONS
# =====================
print("\n" + "="*50)
print("GENERATING VISUALIZATIONS")
print("="*50)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# 1. Player distribution by country
plt.figure(figsize=(12, 6))
country_counts = players_df['country'].value_counts()
plt.bar(country_counts.index, country_counts.values, color='steelblue')
plt.title('Player Distribution by Country', fontsize=16, fontweight='bold')
plt.xlabel('Country', fontsize=12)
plt.ylabel('Number of Players', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/players_by_country.png', dpi=300)
plt.close()
print("✓ Saved: visualizations/players_by_country.png")

# 2. Player distribution by role
plt.figure(figsize=(10, 6))
role_counts = players_df['role'].value_counts()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
plt.pie(role_counts.values, labels=role_counts.index, autopct='%1.1f%%', colors=colors)
plt.title('Player Distribution by Role', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/players_by_role.png', dpi=300)
plt.close()
print("✓ Saved: visualizations/players_by_role.png")

# 3. Age distribution
plt.figure(figsize=(10, 6))
plt.hist(players_df['age'], bins=20, color='coral', edgecolor='black', alpha=0.7)
plt.title('Age Distribution of Players', fontsize=16, fontweight='bold')
plt.xlabel('Age', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.axvline(players_df['age'].mean(), color='red', linestyle='--', label=f"Mean: {players_df['age'].mean():.1f}")
plt.legend()
plt.tight_layout()
plt.savefig('visualizations/age_distribution.png', dpi=300)
plt.close()
print("✓ Saved: visualizations/age_distribution.png")

# 4. Runs scored distribution
plt.figure(figsize=(12, 6))
plt.hist(matches_df['runs_scored'], bins=50, color='green', edgecolor='black', alpha=0.7)
plt.title('Distribution of Runs Scored', fontsize=16, fontweight='bold')
plt.xlabel('Runs', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.axvline(matches_df['runs_scored'].mean(), color='red', linestyle='--', label=f"Mean: {matches_df['runs_scored'].mean():.1f}")
plt.legend()
plt.tight_layout()
plt.savefig('visualizations/runs_distribution.png', dpi=300)
plt.close()
print("✓ Saved: visualizations/runs_distribution.png")

# 5. Wickets taken distribution
plt.figure(figsize=(12, 6))
wickets_data = matches_df[matches_df['wickets_taken'] > 0]
plt.hist(wickets_data['wickets_taken'], bins=7, color='purple', edgecolor='black', alpha=0.7)
plt.title('Distribution of Wickets Taken', fontsize=16, fontweight='bold')
plt.xlabel('Wickets', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/wickets_distribution.png', dpi=300)
plt.close()
print("✓ Saved: visualizations/wickets_distribution.png")

# 6. Match type distribution
plt.figure(figsize=(10, 6))
match_type_counts = matches_df['match_type'].value_counts()
plt.bar(match_type_counts.index, match_type_counts.values, color=['#2E86AB', '#A23B72', '#F18F01'])
plt.title('Match Type Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Match Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/match_type_distribution.png', dpi=300)
plt.close()
print("✓ Saved: visualizations/match_type_distribution.png")

# 7. Team win vs Loss
plt.figure(figsize=(10, 6))
win_counts = matches_df['team_win'].value_counts()
plt.bar(['Loss', 'Win'], win_counts.values, color=['#E74C3C', '#27AE60'])
plt.title('Team Win vs Loss Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Result', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/team_win_distribution.png', dpi=300)
plt.close()
print("✓ Saved: visualizations/team_win_distribution.png")

# 8. Correlation heatmap
plt.figure(figsize=(14, 10))
numeric_cols = matches_df.select_dtypes(include=[np.number]).columns
corr_matrix = matches_df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap - Match Performance', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap.png', dpi=300)
plt.close()
print("✓ Saved: visualizations/correlation_heatmap.png")

# 9. Runs by match type
plt.figure(figsize=(12, 6))
match_type_runs = matches_df.groupby('match_type')['runs_scored'].mean()
plt.bar(match_type_runs.index, match_type_runs.values, color=['#2E86AB', '#A23B72', '#F18F01'])
plt.title('Average Runs by Match Type', fontsize=16, fontweight='bold')
plt.xlabel('Match Type', fontsize=12)
plt.ylabel('Average Runs', fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/runs_by_match_type.png', dpi=300)
plt.close()
print("✓ Saved: visualizations/runs_by_match_type.png")

# 10. Player of match distribution
plt.figure(figsize=(10, 6))
pom_counts = matches_df['player_of_match'].value_counts()
plt.pie(pom_counts.values, labels=['Not Player of Match', 'Player of Match'], 
        autopct='%1.1f%%', colors=['#95A5A6', '#F39C12'])
plt.title('Player of Match Distribution', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/player_of_match_distribution.png', dpi=300)
plt.close()
print("✓ Saved: visualizations/player_of_match_distribution.png")

# 11. Batting style distribution
plt.figure(figsize=(10, 6))
batting_style_counts = players_df['batting_style'].value_counts()
plt.bar(batting_style_counts.index, batting_style_counts.values, color=['#3498DB', '#E91E63'])
plt.title('Batting Style Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Batting Style', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/batting_style_distribution.png', dpi=300)
plt.close()
print("✓ Saved: visualizations/batting_style_distribution.png")

# 12. Experience vs Performance (scatter plot)
# Aggregate player performance
player_agg = matches_df.groupby('player_id').agg({
    'runs_scored': 'sum',
    'wickets_taken': 'sum',
    'team_win': 'sum'
}).reset_index()

player_agg = player_agg.merge(players_df[['player_id', 'experience_years']], on='player_id')

plt.figure(figsize=(12, 6))
plt.scatter(player_agg['experience_years'], player_agg['runs_scored'], 
            alpha=0.6, c='blue', edgecolors='black')
plt.title('Experience vs Total Runs Scored', fontsize=16, fontweight='bold')
plt.xlabel('Experience Years', fontsize=12)
plt.ylabel('Total Runs Scored', fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/experience_vs_runs.png', dpi=300)
plt.close()
print("✓ Saved: visualizations/experience_vs_runs.png")

# =====================
# SAVE CLEANED DATA
# =====================
print("\n" + "="*50)
print("SAVING CLEANED DATA")
print("="*50)

players_df.to_csv("data/players_cleaned.csv", index=False)
matches_df.to_csv("data/matches_cleaned.csv", index=False)

print("✓ Saved: data/players_cleaned.csv")
print("✓ Saved: data/matches_cleaned.csv")

# Create EDA report
eda_report = f"""
# Cricket Performance Data - EDA Report

## Data Overview
- Players: {players_df.shape[0]} records, {players_df.shape[1]} features
- Matches: {matches_df.shape[0]} records, {matches_df.shape[1]} features

## Data Cleaning Performed
1. Filled missing dismissal values with 'not out'
2. Removed duplicate records
3. Converted match_date to datetime format
4. Validated data types

## Key Findings

### Player Demographics
- Countries represented: {players_df['country'].nunique()}
- Most common role: {players_df['role'].mode()[0]}
- Average age: {players_df['age'].mean():.1f} years
- Average experience: {players_df['experience_years'].mean():.1f} years

### Performance Metrics
- Average runs per match: {matches_df['runs_scored'].mean():.2f}
- Average wickets per match: {matches_df['wickets_taken'].mean():.2f}
- Win rate: {matches_df['team_win'].mean()*100:.2f}%
- Player of match rate: {matches_df['player_of_match'].mean()*100:.2f}%

### Match Distribution
- T20 matches: {(matches_df['match_type']=='T20').sum()}
- ODI matches: {(matches_df['match_type']=='ODI').sum()}
- Test matches: {(matches_df['match_type']=='Test').sum()}

## Visualizations Generated
All visualizations saved to: visualizations/

## Next Steps
1. Feature Engineering
2. Model Training
3. Evaluation
"""

with open("eda_reports/eda_summary.md", "w") as f:
    f.write(eda_report)

print("✓ Saved: eda_reports/eda_summary.md")

print("\n" + "="*50)
print("EDA AND DATA CLEANING COMPLETE!")
print("="*50)
