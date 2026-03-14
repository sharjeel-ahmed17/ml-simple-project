"""
Generate synthetic cricket player performance data
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

def generate_cricket_data(n_players=100, n_matches=500):
    """Generate comprehensive cricket performance data"""
    
    # Player base data
    player_names = [
        "Virat Kohli", "Rohit Sharma", "Steve Smith", "Joe Root", "Kane Williamson",
        "Babar Azam", "David Warner", "Ben Stokes", "Jasprit Bumrah", "Pat Cummins",
        "Rashid Khan", "Trent Boult", "Kagiso Rabada", "Shaheen Afridi", "Mitchell Starc",
        "Glenn Maxwell", "Andre Russell", "Hardik Pandya", "Ravindra Jadeja", "R Ashwin",
        "Quinton de Kock", "Jos Buttler", "KL Rahul", "Shubman Gill", "Faf du Plessis",
        "AB de Villiers", "Chris Gayle", "David Miller", "Eoin Morgan", "Aaron Finch"
    ]
    
    # Generate more names if needed
    while len(player_names) < n_players:
        player_names.append(f"Player_{len(player_names)+1}")
    
    player_names = player_names[:n_players]
    
    # Player attributes
    countries = ["India", "Australia", "England", "South Africa", "New Zealand", 
                 "Pakistan", "West Indies", "Sri Lanka", "Bangladesh", "Afghanistan"]
    
    roles = ["Batsman", "Bowler", "All-rounder", "Wicket-keeper"]
    batting_styles = ["Right-hand bat", "Left-hand bat"]
    bowling_styles = ["Right-arm fast", "Left-arm fast", "Right-arm spin", "Left-arm spin", "None"]
    
    players_data = []
    
    for i, name in enumerate(player_names):
        country = random.choice(countries)
        role = random.choice(roles)
        
        # Assign bowling style based on role
        if role == "Batsman":
            bowling_style = random.choice(["None", "Right-arm spin"])
        elif role == "Bowler":
            bowling_style = random.choice(["Right-arm fast", "Left-arm fast", "Right-arm spin", "Left-arm spin"])
        else:
            bowling_style = random.choice(bowling_styles)
        
        age = np.random.randint(18, 40)
        max_exp = max(1, min(age-17, 20))
        experience_years = np.random.randint(1, max_exp + 1)
        
        players_data.append({
            "player_id": i + 1,
            "player_name": name,
            "country": country,
            "role": role,
            "batting_style": random.choice(batting_styles),
            "bowling_style": bowling_style,
            "age": age,
            "experience_years": experience_years
        })
    
    players_df = pd.DataFrame(players_data)
    
    # Match performance data
    matches_data = []
    match_id = 1
    
    venues = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Sydney", 
              "Melbourne", "London", "Manchester", "Dubai", "Cape Town", "Auckland"]
    
    for _ in range(n_matches):
        match_date = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1500))
        venue = random.choice(venues)
        match_type = random.choice(["T20", "ODI", "Test"])
        
        # Select players for this match (11 players per team, 2 teams)
        match_players = random.sample(range(n_players), min(22, n_players))
        
        for player_idx in match_players:
            player = players_data[player_idx]
            player_id = player["player_id"]
            role = player["role"]
            
            # Batting performance
            if role in ["Batsman", "All-rounder", "Wicket-keeper"]:
                balls_faced = np.random.randint(5, 150)
                strike_rate = np.random.uniform(80, 200)
                runs_scored = int((balls_faced * strike_rate) / 100)
                
                # Add some variation
                if random.random() < 0.1:  # 10% chance of duck
                    runs_scored = 0
                    balls_faced = random.randint(1, 5)
                elif random.random() < 0.15:  # 15% chance of big score
                    runs_scored = np.random.randint(50, 150)
                    balls_faced = int(runs_scored * 100 / np.random.uniform(100, 180))
                
                fours = runs_scored // 4 + np.random.randint(-2, 3)
                sixes = runs_scored // 6 + np.random.randint(-2, 3)
                fours = max(0, fours)
                sixes = max(0, sixes)
                
                not_out = 1 if random.random() < 0.2 else 0
                dismissal = None if not_out else random.choice(["bowled", "caught", "lbw", "run out"])
            else:
                runs_scored = 0
                balls_faced = 0
                fours = 0
                sixes = 0
                not_out = 0
                dismissal = None
            
            # Bowling performance
            if role in ["Bowler", "All-rounder"]:
                overs_bowled = np.random.uniform(2, 10)
                economy_rate = np.random.uniform(4, 12)
                runs_conceded = int(overs_bowled * economy_rate)
                wickets_taken = np.random.poisson(2)
                wickets_taken = min(wickets_taken, 6)
                maidens = 1 if economy_rate < 5 and overs_bowled >= 4 else 0
                dots = int(overs_bowled * 6 * np.random.uniform(0.3, 0.6))
            else:
                overs_bowled = 0
                runs_conceded = 0
                wickets_taken = 0
                maidens = 0
                dots = 0
            
            # Fielding performance
            catches = np.random.poisson(0.5)
            run_outs = 1 if random.random() < 0.1 else 0
            stumpings = 1 if role == "Wicket-keeper" and random.random() < 0.2 else 0
            
            # Match result (team win)
            team_win = 1 if random.random() < 0.5 else 0
            
            # Player of match
            player_of_match = 1 if random.random() < 0.05 else 0
            
            matches_data.append({
                "match_id": match_id,
                "player_id": player_id,
                "match_date": match_date.strftime("%Y-%m-%d"),
                "venue": venue,
                "match_type": match_type,
                "opposition": random.choice(countries),
                "runs_scored": runs_scored,
                "balls_faced": balls_faced,
                "fours": fours,
                "sixes": sixes,
                "not_out": not_out,
                "dismissal": dismissal,
                "overs_bowled": round(overs_bowled, 1),
                "runs_conceded": runs_conceded,
                "wickets_taken": wickets_taken,
                "maidens": maidens,
                "dots": dots,
                "catches": catches,
                "run_outs": run_outs,
                "stumpings": stumpings,
                "team_win": team_win,
                "player_of_match": player_of_match
            })
            
            match_id += 1
    
    matches_df = pd.DataFrame(matches_data)
    
    # Save to CSV
    players_df.to_csv("data/players.csv", index=False)
    matches_df.to_csv("data/matches.csv", index=False)
    
    print(f"Generated {len(players_df)} players and {len(matches_df)} match performances")
    print(f"Players data saved to: data/players.csv")
    print(f"Matches data saved to: data/matches.csv")
    
    return players_df, matches_df

if __name__ == "__main__":
    players_df, matches_df = generate_cricket_data()
    print("\nPlayers DataFrame Sample:")
    print(players_df.head())
    print("\nMatches DataFrame Sample:")
    print(matches_df.head())
    print("\nData generation complete!")
