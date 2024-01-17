from model import AdaptationModel
import matplotlib.pyplot as plt
import pandas as pd

# Inizialize the model as you wnat
model = AdaptationModel(number_of_households=1000, flood_map_choice="100yr", network="watts_strogatz", information_policy_type='Knocking',information_policy_radius=8000)

# run 15 steps
for step in range(15):
    model.step()

# Put data in data collector
model_data = model.datacollector.get_model_vars_dataframe()


print("Number of agents per state per step:")
print(model_data[['total_households_state_0', 'total_households_state_1', 'total_households_state_2', 'total_households_state_3', 'total_households_state_4']].to_string())

