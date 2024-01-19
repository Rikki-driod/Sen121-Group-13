import mesa

from model import AdaptationModel
import matplotlib.pyplot as plt
import pandas as pd

# Inizialize the model as you wnat
model = AdaptationModel(number_of_households=1000, flood_map_choice="100yr", network="watts_strogatz",
                        information_policy_type='NL-Alert', information_policy_radius=8000)

# run 15 steps
for step in range(15):
    model.step()

# Put data in data collector
model_data = model.datacollector.get_model_vars_dataframe()

print("Number of agents per state per step:")
print(model_data[['total_households_state_0', 'total_households_state_1', 'total_households_state_2',
                  'total_households_state_3', 'total_households_state_4']].to_string())

params = {"number_of_households": 100, "information_policy_radius": 8000,
          "flood_map_choice": ['100yr', '500yr', 'harvey'],
          "network": ['watts_strogatz', 'erdos_renyi', 'barabasi_albert'],
          "information_policy_type": ['Knocking', 'NL-Alert']}

results = mesa.batch_run(
    AdaptationModel,
    parameters=params,
    iterations=1,
    max_steps=14,
    number_processes=1,
    data_collection_period=1,
    display_progress=True,
)

results_df=pd.DataFrame(results, columns=['iteration', 'Step', 'flood_map_choice', 'network', 'information_policy_type', 'total_households_state_0', 'total_households_state_1', 'total_households_state_2', 'total_households_state_3', 'total_households_state_4'])


print(results_df.keys())

#results_df.filter(items=['iteration', 'flood_map_choice', 'network'])

#, 'information_policy_type', 'total_households_state_0', 'total_households_state_1', 'total_households_state_2', 'total_households_state_3', 'total_households_state_4'])

#print(results_df)

#results_filtered = results_df[(results_df.AgentID == 0) & (results_df.Step == 100)]
#results_filtered[["iteration", "flood_map_choice"]].reset_index(
 #   drop=True
#).head()

#print(results_filtered)

results_df.to_excel(r'C:\Users\RDvan\OneDrive\Documents\GitHub\Sen121-Group-13\output.xlsx', index=False)



# results_df_filtered = results_df[(results_df.AgentID == 0) & (results_df.Step == 15) & (results_df.number_of_households == 1000)]
# results_df_filtered.head(3)

# print(results_df_filtered)
