import mesa
import seaborn

from model import AdaptationModel
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Inizialize the model as you wnat
model = AdaptationModel(number_of_households=1000, flood_map_choice="100yr", network="watts_strogatz",
                        information_policy_type='NL-Alert', information_policy_radius=8000)

# run 15 steps
for step in range(15):
    model.step()

# Put data in data collector
model_data = model.datacollector.get_model_vars_dataframe()

print("Number of agents per state per step:")
print(model_data[['total_HH_state_0', 'total_HH_state_1', 'total_HH_state_2',
                  'total_HH_state_3', 'total_HH_state_4']].to_string())

params = {"number_of_households": 1000, "information_policy_radius": 8000,
          "flood_map_choice": ['100yr', '500yr', 'harvey'],
          "network": ['watts_strogatz', 'erdos_renyi', 'barabasi_albert'],
          "information_policy_type": ['Knocking', 'NL-Alert']}

results = mesa.batch_run(
    AdaptationModel,
    parameters=params,
    iterations=25,
    max_steps=14,
    number_processes=1,
    data_collection_period=1,
    display_progress=True,
)

results_df = pd.DataFrame(results) #, columns=['iteration', 'Step', 'AgentID', 'flood_map_choice', 'network',
                                           # 'information_policy_type', 'total_households_state_0',
                                            #'total_households_state_1', 'total_households_state_2',
                                            #'total_households_state_3', 'total_households_state_4',
                                            #'flood_damage_actual'])

results_filtered = results_df[(results_df.AgentID == 0)]

print(results_filtered.keys())

results_filtered.to_excel(r'C:\Users\RDvan\OneDrive\Documents\GitHub\Sen121-Group-13\output.xlsx', index=False)

g_4 = sns.lineplot(
    data=results_filtered,
    x="Step",
    y="total_HH_state_4",
    hue="information_policy_type",
    errorbar=("ci", 95),
    palette="tab10",
)
g_4.figure.set_size_inches(8, 4)
plot_title = "Household final state growth over time"
g_4.set(title=plot_title, ylabel="Total number of Households", xlabel="Step");


g = sns.PairGrid(results_filtered, y_vars=["total_HH_state_0",
                                           "total_HH_state_1", "total_HH_state_2",
                                           "total_HH_state_3"], x_vars=["Step"],
                 hue="information_policy_type", height=4)
g.map(sns.lineplot)
g.add_legend()

results_filtered_2 = results_df[(results_df.AgentID == 0) & (results_df.Step == 14)]

scatter_map = sns.PairGrid(results_filtered_2, y_vars=["average_actual_flood_damage",
                                                     "average_init_flood_damage"],
                           x_vars=["RunId"], hue="information_policy_type", height=4)
scatter_map.map(sns.scatterplot)
scatter_map.add_legend()

scatter_map_2 = sns.PairGrid(results_filtered_2, y_vars=["average_actual_flood_damage",
                                                     "average_init_flood_damage"],
                           x_vars=["RunId"], hue="flood_map_choice", height=4)
scatter_map_2.map(sns.scatterplot)
scatter_map_2.add_legend()

g_bar = sns.PairGrid(results_filtered_2, y_vars=["total_HH_state_4",
                                           ], x_vars=["network"],
                 hue="information_policy_type", height=4)
g_bar.map(sns.barplot)
g_bar.add_legend()

plt.show()




# concatenated = pd.concat([results_filtered.total_households_state_0.assign(dataset='set1'),
# results_filtered.total_households_state_1.assign(dataset='set2'), results_filtered.total_households_state_2.assign(
# dataset='set3'), results_filtered.total_households_state_3.assign(dataset='set4'),
# results_filtered.total_households_state_4.assign(dataset='set5')])

# sns.scatterplot(x='iteration', y='Step', data=concatenated)


# class v / s fare barplot
# sns.set_style('whitegrid')
# sns.barplot(x='iteration', y='total_households_state_3', data=results_filtered)

# Show the plot
# plt.show()

# seaborn.set(style='whitegrid')

# seaborn.scatterplot(x="Step",
# y="total_households_state_2",
# data=results_filtered)

# plt.show()


# results_df_filtered = results_df[(results_df.AgentID == 0) & (results_df.Step == 15) & (
# results_df.number_of_households == 1000)] results_df_filtered.head(3)

# print(results_df_filtered)

# results_df.filter(items=['iteration', 'flood_map_choice', 'network'])

# , 'information_policy_type', 'total_households_state_0', 'total_households_state_1', 'total_households_state_2', 'total_households_state_3', 'total_households_state_4'])

# print(results_df)


# results_filtered[["iteration", "flood_map_choice"]].reset_index(
#   drop=True
# ).head()

# print(results_filtered)

# g_0 = sns.lineplot(
#    data=results_filtered,
#    x="Step",
#    y="total_households_state_0",
#    hue="information_policy_type",
#    errorbar=("ci", 95),
#    palette="tab10",
# )
# g_0.figure.set_size_inches(8, 4)
# plot_title = "Household final state (0) growth over time"
# g_0.set(title=plot_title, ylabel="Total number of Households", xlabel="Step");

# g_1 = sns.lineplot(
#    data=results_filtered,
#    x="Step",
#    y="total_households_state_1",
#    hue="information_policy_type",
#    errorbar=("ci", 95),
#    palette="tab10",
# )
# g_1.figure.set_size_inches(8, 4)
# plot_title = "Household final state (1) growth over time"
# g_1.set(title=plot_title, ylabel="Total number of Households", xlabel="Step");

# g_2 = sns.lineplot(
#    data=results_filtered,
#    x="Step",
#    y="total_households_state_2",
#    hue="information_policy_type",
#    errorbar=("ci", 95),
#    palette="tab10",
# )
# g_2.figure.set_size_inches(8, 4)
# plot_title = "Household final state (2) growth over time"
# g_2.set(title=plot_title, ylabel="Total number of Households", xlabel="Step");

# g_3 = sns.lineplot(
#    data=results_filtered,
#    x="Step",
#    y="total_households_state_3",
#    hue="information_policy_type",
#    errorbar=("ci", 95),
#    palette="tab10",
# )
# g_3.figure.set_size_inches(8, 4)
# plot_title = "Household final state (3) growth over time"
# g_3.set(title=plot_title, ylabel="Total number of Households", xlabel="Step");

