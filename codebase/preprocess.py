# import kagglehub
import json
import math

# # Download latest version
# path = kagglehub.dataset_download("zynicide/nfl-football-player-stats")
# path2 = kagglehub.dataset_download("nicholasliusontag/nfl-contract-and-draft-data")


# print("Path to dataset files:", path)
# print("Path to dataset files:", path2)

games_stats = open("/Users/mratmeyer/.cache/kagglehub/datasets/zynicide/nfl-football-player-stats/versions/1/games_1512362753.8735218.json")
games_json = json.load(games_stats)

profiles_stats = open("/Users/mratmeyer/.cache/kagglehub/datasets/zynicide/nfl-football-player-stats/versions/1/profiles_1512362725.022629.json")
profiles_json = json.load(profiles_stats)

contracts_data = open("/Users/mratmeyer/.cache/kagglehub/datasets/nicholasliusontag/nfl-contract-and-draft-data/versions/2/combined_data_2000-2023.csv", "r")

### Year range processing

min_year = dict()
contract_years = dict()

for i, line in enumerate(contracts_data):
    if i == 0:
        continue

    fields = line.split(',')
    contract_player_name = fields[4]
    contract_position = fields[5]
    contract_year = math.floor(float(fields[10]))

    min_year[contract_player_name] = min(contract_year, min_year.get(contract_player_name, float('inf')))
    if contract_player_name not in contract_years:
        contract_years[contract_player_name] = []
    if contract_year not in contract_years[contract_player_name]:
        contract_years[contract_player_name].append(contract_year)

year_ranges = dict()

for player in contract_years:
    year_ranges[player] = dict()

    contract_years[player].sort()

    for i in range(len(contract_years[player]) - 1):
        current_min_year = contract_years[player][i]
        current_range = [current_min_year]
        current_year = current_min_year + 1

        while current_year not in contract_years[player]:
            current_range.append(current_year)
            current_year += 1
        
        year_ranges[player][contract_years[player][i + 1]] = current_range

### End year range processing

games_stats = open("/Users/mratmeyer/.cache/kagglehub/datasets/zynicide/nfl-football-player-stats/versions/1/games_1512362753.8735218.json")
games_json = json.load(games_stats)

profiles_stats = open("/Users/mratmeyer/.cache/kagglehub/datasets/zynicide/nfl-football-player-stats/versions/1/profiles_1512362725.022629.json")
profiles_json = json.load(profiles_stats)

contracts_data = open("/Users/mratmeyer/.cache/kagglehub/datasets/nicholasliusontag/nfl-contract-and-draft-data/versions/2/combined_data_2000-2023.csv", "r")

f = open("output.csv", "a")
f.write("contract_player_name,contract_year,contract_value,total_passing_attempts,total_passing_completions,total_passing_yards,total_passing_rating,total_passing_touchdowns,total_passing_interceptions,total_passing_sacks,total_passing_sacks_yards_lost,total_rushing_attempts,total_rushing_yards,total_rushing_touchdowns,total_games\n")

for i, line in enumerate(contracts_data):
    if i == 0:
        continue

    fields = line.split(',')
    contract_player_name = fields[4]
    contract_position = fields[5]
    contract_value = fields[12]
    contract_year = math.floor(float(fields[10]))

    for profile in profiles_json:
        if contract_player_name == profile['name'] and contract_position == profile['position'] and profile['position'] == 'QB':
            total_passing_attempts = 0
            total_passing_completions = 0
            total_passing_yards = 0
            total_passing_rating = 0
            total_passing_touchdowns = 0
            total_passing_interceptions = 0
            total_passing_sacks = 0
            total_passing_sacks_yards_lost = 0
            total_rushing_attempts = 0
            total_rushing_yards = 0
            total_rushing_touchdowns = 0

            total_games = 0

            for game_player in games_json:
                if profile['player_id'] == game_player['player_id'] and (contract_year in year_ranges[profile['name']] and int(game_player['year']) in year_ranges[profile['name']][contract_year]):
                    passing_attempts = game_player['passing_attempts']
                    passing_completions = game_player['passing_completions']
                    passing_yards = game_player['passing_yards']
                    passing_rating = game_player['passing_rating']
                    passing_touchdowns = game_player['passing_touchdowns']
                    passing_interceptions = game_player['passing_interceptions']
                    passing_sacks = game_player['passing_sacks']
                    passing_sacks_yards_lost = game_player['passing_sacks_yards_lost']
                    rushing_attempts = game_player['rushing_attempts']
                    rushing_yards = game_player['rushing_yards']
                    rushing_touchdowns = game_player['rushing_touchdowns']

                    total_passing_attempts += passing_attempts
                    total_passing_completions += passing_completions
                    total_passing_yards += passing_yards
                    total_passing_rating += passing_rating
                    total_passing_touchdowns += passing_touchdowns
                    total_passing_interceptions += passing_interceptions
                    total_passing_sacks += passing_sacks
                    total_passing_sacks_yards_lost += passing_sacks_yards_lost
                    total_rushing_attempts += rushing_attempts
                    total_rushing_yards += rushing_yards
                    total_rushing_touchdowns += rushing_touchdowns
                    total_games += 1
            
            data_line = f"{contract_player_name},{contract_year},{contract_value},{total_passing_attempts},{total_passing_completions},{total_passing_yards},{total_passing_rating},{total_passing_touchdowns},{total_passing_interceptions},{total_passing_sacks},{total_passing_sacks_yards_lost},{total_rushing_attempts},{total_rushing_yards},{total_rushing_touchdowns},{total_games}\n"
            f.write(data_line)

f.close()