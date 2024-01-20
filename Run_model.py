import mesa
import seaborn

from model import AdaptationModel
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



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

params = {"number_of_households": 1000, "information_policy_radius": 8000,
          "flood_map_choice": ['100yr', '500yr', 'harvey'],
          "network": ['watts_strogatz', 'erdos_renyi', 'barabasi_albert'],
          "information_policy_type": ['Knocking', 'NL-Alert']}

results = mesa.batch_run(
    AdaptationModel,
    parameters=params,
    iterations=3,
    max_steps=14,
    number_processes=1,
    data_collection_period=1,
    display_progress=True,
)



results_df=pd.DataFrame(results, columns=['iteration', 'Step', 'AgentID', 'flood_map_choice', 'network', 'information_policy_type', 'total_households_state_0', 'total_households_state_1', 'total_households_state_2', 'total_households_state_3', 'total_households_state_4'])

results_filtered = results_df[(results_df.AgentID == 0)]

print(results_filtered.keys())

#results_df.filter(items=['iteration', 'flood_map_choice', 'network'])

#, 'information_policy_type', 'total_households_state_0', 'total_households_state_1', 'total_households_state_2', 'total_households_state_3', 'total_households_state_4'])

#print(results_df)


#results_filtered[["iteration", "flood_map_choice"]].reset_index(
 #   drop=True
#).head()

#print(results_filtered)

results_filtered.to_excel(r'C:\Users\RDvan\OneDrive\Documents\GitHub\Sen121-Group-13\output.xlsx', index=False)

g = sns.lineplot(
    data=results_filtered,
    x="Step",
    y="total_households_state_4",
    hue="information_policy_type",
    errorbar=("ci", 95),
    palette="tab10",
)
g.figure.set_size_inches(8, 4)
plot_title = "Household final state growth over time"
g.set(title=plot_title, ylabel="Total number of Households", xlabel="Step");

plt.show()

#concatenated = pd.concat([results_filtered.total_households_state_0.assign(dataset='set1'), results_filtered.total_households_state_1.assign(dataset='set2'), results_filtered.total_households_state_2.assign(dataset='set3'), results_filtered.total_households_state_3.assign(dataset='set4'), results_filtered.total_households_state_4.assign(dataset='set5')])

#sns.scatterplot(x='iteration', y='Step', data=concatenated)


# class v / s fare barplot
#sns.set_style('whitegrid')
#sns.barplot(x='iteration', y='total_households_state_3', data=results_filtered)

# Show the plot
#plt.show()

#seaborn.set(style='whitegrid')

#seaborn.scatterplot(x="Step",
                    #y="total_households_state_2",
                   # data=results_filtered)

#plt.show()


# results_df_filtered = results_df[(results_df.AgentID == 0) & (results_df.Step == 15) & (results_df.number_of_households == 1000)]
# results_df_filtered.head(3)

# print(results_df_filtered)
