"""
Streamlit UI for Cricket Performance Prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
os.chdir(project_root)

# Page configuration
st.set_page_config(
    page_title="Cricket Performance Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .prediction-high {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .prediction-low {
        background-color: #f8d7da;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_models():
    """Load ML models and preprocessors"""
    model_path = Path("models")
    
    try:
        model = joblib.load(model_path / "best_model_Gradient_Boosting.joblib")
        scaler = joblib.load(model_path / "scaler.joblib")
        encoders = joblib.load(model_path / "label_encoders.joblib")
        
        with open(model_path / "feature_columns.json", "r") as f:
            feature_columns = json.load(f)
        
        with open(model_path / "model_metadata.json", "r") as f:
            metadata = json.load(f)
        
        return model, scaler, encoders, feature_columns, metadata
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

# Load EDA data
@st.cache_data
def load_data():
    """Load cleaned data for visualizations"""
    try:
        players_df = pd.read_csv("data/players_cleaned.csv")
        matches_df = pd.read_csv("data/matches_cleaned.csv")
        return players_df, matches_df
    except:
        return None, None

model, scaler, encoders, feature_columns, metadata = load_models()
players_df, matches_df = load_data()

# Header
st.markdown('<p class="main-header">🏏 Cricket Performance Predictor</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.header("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🎯 Prediction", "📈 Analytics Dashboard", "ℹ️ About"]
)

st.sidebar.markdown("---")

if metadata:
    st.sidebar.subheader("🤖 Model Info")
    st.sidebar.metric("Model", metadata.get("best_model", "N/A"))
    st.sidebar.metric("Accuracy", f"{metadata.get('accuracy', 0)*100:.2f}%")
    st.sidebar.metric("AUC-ROC", f"{metadata.get('auc_roc', 0)*100:.2f}%")

# =====================
# PREDICTION PAGE
# =====================
if page == "🎯 Prediction":
    st.header("Predict Player Performance")
    st.write("Enter match statistics to predict if a player will have a high performance.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Player & Match Details")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=50, value=28)
            experience = st.number_input("Experience (Years)", min_value=0, max_value=30, value=5)
            role = st.selectbox("Role", ["Batsman", "Bowler", "All-rounder", "Wicket-keeper"])
        
        with c2:
            batting_style = st.selectbox("Batting Style", ["Right-hand bat", "Left-hand bat"])
            bowling_style = st.selectbox("Bowling Style", [
                "Right-arm fast", "Left-arm fast", "Right-arm spin", "Left-arm spin", "None"
            ])
            match_type = st.selectbox("Match Type", ["T20", "ODI", "Test"])
        
        with c3:
            venue = st.selectbox("Venue", [
                "Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata",
                "Sydney", "Melbourne", "London", "Manchester", "Dubai"
            ])
            opposition = st.selectbox("Opposition", [
                "India", "Australia", "England", "South Africa", "New Zealand",
                "Pakistan", "West Indies", "Sri Lanka", "Bangladesh", "Afghanistan"
            ])
        
        st.subheader("🏏 Batting Statistics")
        b1, b2, b3, b4, b5 = st.columns(5)
        with b1:
            balls_faced = st.number_input("Balls Faced", min_value=0, value=50)
        with b2:
            runs_scored = st.number_input("Runs Scored", min_value=0, value=60)
        with b3:
            fours = st.number_input("Fours", min_value=0, value=6)
        with b4:
            sixes = st.number_input("Sixes", min_value=0, value=2)
        with b5:
            not_out = st.selectbox("Not Out", [0, 1])
        
        st.subheader("🎾 Bowling Statistics")
        bo1, bo2, bo3, bo4, bo5 = st.columns(5)
        with bo1:
            overs_bowled = st.number_input("Overs Bowled", min_value=0.0, value=4.0)
        with bo2:
            runs_conceded = st.number_input("Runs Conceded", min_value=0, value=30)
        with bo3:
            wickets_taken = st.number_input("Wickets Taken", min_value=0, max_value=10, value=2)
        with bo4:
            maidens = st.number_input("Maiden Overs", min_value=0, value=0)
        with bo5:
            dots = st.number_input("Dot Balls", min_value=0, value=12)
        
        st.subheader("🙌 Fielding Statistics")
        f1, f2, f3 = st.columns(3)
        with f1:
            catches = st.number_input("Catches", min_value=0, value=1)
        with f2:
            run_outs = st.number_input("Run Outs", min_value=0, value=0)
        with f3:
            stumpings = st.number_input("Stumpings", min_value=0, value=0)
    
    with col2:
        st.subheader("📊 Quick Stats")
        
        # Calculate strike rate
        strike_rate = (runs_scored / balls_faced * 100) if balls_faced > 0 else 0
        economy = (runs_conceded / overs_bowled) if overs_bowled > 0 else 0
        
        st.metric("Strike Rate", f"{strike_rate:.1f}")
        st.metric("Economy Rate", f"{economy:.2f}")
        st.metric("Boundary Count", fours + sixes)
        st.metric("Total Contributions", catches + run_outs + stumpings + wickets_taken)
        
        st.markdown("---")
        
        # Predict button
        predict_btn = st.button("🔮 Predict Performance", type="primary", use_container_width=True)
        
        if predict_btn and model is not None:
            # Prepare input data
            input_data = {
                'age': age,
                'experience_years': experience,
                'balls_faced': balls_faced,
                'runs_scored': runs_scored,
                'fours': fours,
                'sixes': sixes,
                'not_out': not_out,
                'overs_bowled': overs_bowled,
                'runs_conceded': runs_conceded,
                'wickets_taken': wickets_taken,
                'maidens': maidens,
                'dots': dots,
                'catches': catches,
                'run_outs': run_outs,
                'stumpings': stumpings,
                'match_type': match_type,
                'role': role,
                'batting_style': batting_style,
                'bowling_style': bowling_style,
                'venue': venue,
                'opposition': opposition
            }
            
            # Calculate derived features
            input_data['strike_rate'] = (runs_scored / balls_faced * 100) if balls_faced > 0 else 0
            input_data['boundary_percentage'] = ((fours + sixes) / balls_faced * 100) if balls_faced > 0 else 0
            input_data['runs_per_boundary'] = (runs_scored / (fours + sixes)) if (fours + sixes) > 0 else runs_scored
            input_data['economy_rate'] = (runs_conceded / overs_bowled) if overs_bowled > 0 else 0
            input_data['wickets_per_over'] = (wickets_taken / overs_bowled) if overs_bowled > 0 else 0
            input_data['dot_ball_percentage'] = ((dots / (overs_bowled * 6)) * 100) if overs_bowled > 0 else 0
            input_data['total_fielding_contributions'] = catches + run_outs + stumpings
            input_data['batting_performance_score'] = (
                runs_scored * 1.0 + fours * 2 + sixes * 3 + not_out * 10 + input_data['strike_rate'] * 0.1
            )
            input_data['bowling_performance_score'] = (
                wickets_taken * 25 + maidens * 10 + dots * 2 - runs_conceded * 0.5
            )
            input_data['match_importance'] = {'Test': 3, 'ODI': 2, 'T20': 1}.get(match_type, 1)
            input_data['is_home_venue'] = 0
            
            # Encode categorical
            try:
                input_data['role_encoded'] = encoders['role'].transform([role])[0]
                input_data['batting_style_encoded'] = encoders['batting_style'].transform([batting_style])[0]
                input_data['bowling_style_encoded'] = encoders['bowling_style'].transform([bowling_style])[0]
                input_data['match_type_encoded'] = encoders['match_type'].transform([match_type])[0]
                input_data['venue_encoded'] = encoders['venue'].transform([venue])[0]
            except:
                input_data['role_encoded'] = 0
                input_data['batting_style_encoded'] = 0
                input_data['bowling_style_encoded'] = 0
                input_data['match_type_encoded'] = 0
                input_data['venue_encoded'] = 0
            
            input_data['experience_level_encoded'] = 0
            input_data['age_group_encoded'] = 0
            
            # Create feature vector
            feature_vector = [input_data.get(col, 0) for col in feature_columns]
            feature_array = np.array(feature_vector).reshape(1, -1)
            feature_scaled = scaler.transform(feature_array)
            
            # Predict
            prediction = model.predict(feature_scaled)[0]
            probability = model.predict_proba(feature_scaled)[0][1]
            
            # Display result
            st.markdown("---")
            st.subheader("🎯 Prediction Result")
            
            if prediction == 1:
                st.markdown(f"""
                    <div class="prediction-high">
                        <h3>✅ High Performance Predicted!</h3>
                        <p>The player is likely to have a <strong>high performance</strong> in this match.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="prediction-low">
                        <h3>📉 Normal/Low Performance Predicted</h3>
                        <p>The player is likely to have a <strong>normal/low performance</strong> in this match.</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Probability gauge
            st.markdown("#### Confidence Level")
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=probability * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "High Performance Probability", 'font': {'size': 16}},
                delta={'reference': 50, 'increasing': {'color': "RebeccaPurple"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': '#f8d7da'},
                        {'range': [50, 75], 'color': '#fff3cd'},
                        {'range': [75, 100], 'color': '#d4edda'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=250, margin={'l': 20, 'r': 20, 't': 40, 'b': 20})
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"📊 Model Confidence: **{probability*100:.1f}%**")
    
    # Sample predictions
    st.markdown("---")
    st.subheader("🎲 Try Sample Scenarios")
    
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        if st.button("💥 Century Maker", use_container_width=True):
            st.session_state.century = True
            st.session_state.economy = False
            st.session_state.balanced = False
    
    with col_s2:
        if st.button("🎯 Bowler's Dream", use_container_width=True):
            st.session_state.century = False
            st.session_state.economy = True
            st.session_state.balanced = False
    
    with col_s3:
        if st.button("⚖️ All-Rounder", use_container_width=True):
            st.session_state.century = False
            st.session_state.economy = False
            st.session_state.balanced = True
    
    if st.session_state.get('century', False):
        st.success("Loaded: Century scenario - 100 runs, 10 fours, 4 sixes!")
        balls_faced = 80
        runs_scored = 100
        fours = 10
        sixes = 4
        wickets_taken = 0
        overs_bowled = 0
    
    if st.session_state.get('economy', False):
        st.success("Loaded: Bowler scenario - 5 wickets, 2 maidens, economy 4.5!")
        overs_bowled = 10
        runs_conceded = 45
        wickets_taken = 5
        maidens = 2
        dots = 40
        runs_scored = 10
        balls_faced = 15
    
    if st.session_state.get('balanced', False):
        st.success("Loaded: All-rounder scenario - 50 runs + 3 wickets!")
        balls_faced = 40
        runs_scored = 50
        fours = 5
        sixes = 2
        overs_bowled = 8
        runs_conceded = 50
        wickets_taken = 3
        maidens = 1
        dots = 30

# =====================
# ANALYTICS DASHBOARD
# =====================
elif page == "📈 Analytics Dashboard":
    st.header("📊 Analytics Dashboard")
    st.write("Explore cricket performance data and insights.")
    
    if players_df is not None and matches_df is not None:
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Players", players_df.shape[0])
        with col2:
            st.metric("Total Matches", matches_df['match_id'].nunique())
        with col3:
            st.metric("Match Performances", matches_df.shape[0])
        with col4:
            st.metric("Avg Runs/Match", f"{matches_df['runs_scored'].mean():.1f}")
        
        st.markdown("---")
        
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["🏏 Player Stats", "📈 Performance", "🌍 Geography", "🎯 Distributions"])
        
        with tab1:
            st.subheader("Player Distribution by Role")
            role_counts = players_df['role'].value_counts().reset_index()
            role_counts.columns = ['Role', 'Count']
            
            fig = px.pie(role_counts, values='Count', names='Role', hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Top Players by Total Runs")
            player_runs = matches_df.groupby('player_id')['runs_scored'].sum().reset_index()
            player_runs = player_runs.sort_values('runs_scored', ascending=False).head(10)
            
            fig = px.bar(player_runs, x='player_id', y='runs_scored', 
                        labels={'player_id': 'Player ID', 'runs_scored': 'Total Runs'},
                        color='runs_scored', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Runs Distribution")
            fig = px.histogram(matches_df, x='runs_scored', nbins=50,
                             labels={'runs_scored': 'Runs Scored'},
                             color_discrete_sequence=['#1E88E5'])
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Wickets Distribution")
            wickets_data = matches_df[matches_df['wickets_taken'] > 0]
            fig = px.histogram(wickets_data, x='wickets_taken', nbins=10,
                             labels={'wickets_taken': 'Wickets Taken'},
                             color_discrete_sequence=['#E53935'])
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Players by Country")
            country_counts = players_df['country'].value_counts().reset_index()
            country_counts.columns = ['Country', 'Players']
            
            fig = px.bar(country_counts, x='Country', y='Players',
                        color='Players', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Match Venues")
            venue_counts = matches_df['venue'].value_counts().head(10).reset_index()
            venue_counts.columns = ['Venue', 'Matches']
            
            fig = px.bar(venue_counts, x='Venue', y='Matches',
                        color='Matches', color_continuous_scale='Oranges')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Match Type Distribution")
            match_type_counts = matches_df['match_type'].value_counts().reset_index()
            match_type_counts.columns = ['Type', 'Count']
            
            fig = px.bar(match_type_counts, x='Type', y='Count',
                        color='Count', color_discrete_sequence=['#43A047', '#FDD835', '#FB8C00'])
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Win vs Loss Distribution")
            win_counts = matches_df['team_win'].value_counts().reset_index()
            win_counts.columns = ['Result', 'Count']
            win_counts['Result'] = win_counts['Result'].map({0: 'Loss', 1: 'Win'})
            
            fig = px.pie(win_counts, values='Count', names='Result',
                        color='Result', color_discrete_map={'Win': '#28a745', 'Loss': '#dc3545'})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Data files not found. Please run the data generation and EDA scripts first.")

# =====================
# ABOUT PAGE
# =====================
elif page == "ℹ️ About":
    st.header("About This Project")
    
    st.markdown("""
    ### 🏏 Cricket Performance Predictor
    
    This application uses machine learning to predict cricket player performance based on match statistics.
    
    #### Features
    - **🎯 Performance Prediction**: Predict if a player will have high or normal/low performance
    - **📊 Analytics Dashboard**: Explore data visualizations and insights
    - **🤖 ML Model**: Gradient Boosting Classifier with 99.6% accuracy
    
    #### How It Works
    1. Enter player and match details
    2. Input batting, bowling, and fielding statistics
    3. Get instant performance prediction with confidence score
    
    #### Model Features
    - Player attributes (age, experience, role)
    - Batting statistics (runs, strike rate, boundaries)
    - Bowling statistics (wickets, economy, maidens)
    - Fielding contributions (catches, run-outs, stumpings)
    - Match context (type, venue, opposition)
    
    #### Technology Stack
    - **Backend**: FastAPI
    - **Frontend**: Streamlit
    - **ML**: Scikit-learn (Gradient Boosting)
    - **Visualization**: Plotly
    
    ---
    
    **Model Performance:**
    """)
    
    if metadata:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{metadata.get('accuracy', 0)*100:.2f}%")
        with col2:
            st.metric("AUC-ROC", f"{metadata.get('auc_roc', 0)*100:.2f}%")
        with col3:
            st.metric("Features", metadata.get('feature_count', 0))
    
    st.markdown("""
    
    ---
    
    **Created with ❤️ for Cricket Analytics**
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>🏏 Cricket Performance Predictor | Built with FastAPI & Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
