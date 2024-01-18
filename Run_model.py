import mesa

from model import AdaptationModel
import matplotlib.pyplot as plt
import pandas as pd

# Inizialize the model as you wnat
model = AdaptationModel(number_of_households=1000, flood_map_choice="100yr", network="watts_strogatz", information_policy_type='NL-Alert',information_policy_radius=8000)

# run 15 steps
for step in range(15):
    model.step()

# Put data in data collector
model_data = model.datacollector.get_model_vars_dataframe()



print("Number of agents per state per step:")
print(model_data[['total_households_state_0', 'total_households_state_1', 'total_households_state_2', 'total_households_state_3', 'total_households_state_4']].to_string())

params = {"number_of_households": 1000, "information_policy_radius": 8000, "flood_map_choice": ['100yr', '500yr', 'harvey'], "network": ['watts_strogatz', 'erdos_renyi', 'barabasi_albert'], "information_policy_type": ['knocking', 'NL-Alert']}

results = mesa.batch_run(
    AdaptationModel,
    parameters=params,
    iterations=3,
    max_steps=45,
    number_processes=1,
    data_collection_period=15,
    display_progress=True,
)

results_df=pd.DataFrame(results)

print(results_df.keys())

#results_df.to_excel('output.xlsx', index=False)


