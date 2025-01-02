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

print(year_ranges['Courtney Brown'])
