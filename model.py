# Importing necessary libraries
import networkx as nx
from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import geopandas as gpd
import rasterio as rs
import matplotlib.pyplot as plt
import random

# Import the agent class(es) from agents.py
from agents import Households

# Import functions from functions.py
from functions import get_flood_map_data, calculate_basic_flood_damage
from functions import map_domain_gdf, floodplain_gdf


# Define the AdaptationModel class
class AdaptationModel(Model):
    """
    The main model running the simulation. It sets up the network of household agents,
    simulates their behavior, and collects data. The network type can be adjusted based on study requirements.
    """

    def __init__(self,
                 seed=None,
                 number_of_households=1000,  # number of household agents
                 # Simplified argument for choosing flood map. Can currently be "harvey", "100yr", or "500yr".
                 flood_map_choice='harvey',
                 # ### network related parameters ###
                 # The social network structure that is used.
                 # Can currently be "erdos_renyi", "barabasi_albert", "watts_strogatz", or "no_network"
                 network='watts_strogatz',
                 # likeliness of edge being created between two nodes
                 probability_of_network_connection=0.4,
                 # number of edges for BA network
                 number_of_edges=3,
                 # number of nearest neighbours for WS social network
                 number_of_nearest_neighbours=5,
                 information_policy_type = 'knocking',
                 information_policy_radius=None  # New parameter for radius
    ):

        super().__init__(seed=seed)

        # defining the variables and setting the values
        self.number_of_households = number_of_households  # Total number of household agents
        self.seed = seed
        #information policy type
        self.information_policy_type = information_policy_type
        self.information_policy_radius = information_policy_radius
        # network
        self.network = network  # Type of network to be created
        self.probability_of_network_connection = probability_of_network_connection
        self.number_of_edges = number_of_edges
        self.number_of_nearest_neighbours = number_of_nearest_neighbours

        # generating the graph according to the network used and the network parameters specified
        self.G = self.initialize_network()
        # create grid out of network graph
        self.grid = NetworkGrid(self.G)

        # Initialize maps
        self.initialize_maps(flood_map_choice)

        # set schedule for agents
        self.schedule = RandomActivation(self)  # Schedule for activating agents

        # create households through initiating a household on each node of the network graph
        for i, node in enumerate(self.G.nodes()):
            household = Households(unique_id=i, model=self)
            self.schedule.add(household)
            self.grid.place_agent(agent=household, node_id=node)



        # Data collection setup to collect data
        model_metrics = {
            "total_households_state_0": lambda m: sum(1 for agent in m.schedule.agents if agent.state == 0),
            "total_households_state_1": lambda m: sum(1 for agent in m.schedule.agents if agent.state == 1),
            "total_households_state_2": lambda m: sum(1 for agent in m.schedule.agents if agent.state == 2),
            "total_households_state_3": lambda m: sum(1 for agent in m.schedule.agents if agent.state == 3),
            "total_households_state_4": lambda m: sum(1 for agent in m.schedule.agents if agent.state == 4),
            "average_actual_flood_depth": lambda m: sum(agent.flood_depth_actual for agent in m.schedule.agents) / len(
                m.schedule.agents),
            "average_actual_flood_damage": lambda m: sum(
                agent.flood_damage_actual for agent in m.schedule.agents) / len(m.schedule.agents),
            "proportion_reached_by_info_policy": lambda m: sum(1 for agent in m.schedule.agents if agent.reached) / len(
                m.schedule.agents),
            "proportion_in_floodplain": lambda m: sum(1 for agent in m.schedule.agents if agent.in_floodplain) / len(
                m.schedule.agents),
            "average_level_of_knowledge": lambda m: sum(agent.level_of_knowledge for agent in m.schedule.agents) / len(
                m.schedule.agents),
            # Add or modify other metrics as needed
        }

        agent_metrics = {
            "FloodDepthActual": "flood_depth_actual",
            "FloodDamageActual": "flood_damage_actual",
            "State": "state",
            "InFloodplain": "in_floodplain",
            "LevelOfKnowledge": "level_of_knowledge",
            "HasBeenReached": "reached",
            "Location": lambda a: (a.location.x, a.location.y)  # Assuming location is a Point object
            # Add or modify other agent-specific metrics as needed
        }
        # set up the data collector
        self.datacollector = DataCollector(model_reporters=model_metrics, agent_reporters=agent_metrics)

    def initialize_network(self):
        """
        Initialize and return the social network graph based on the provided network type using pattern matching.
        """
        if self.network == 'erdos_renyi':
            return nx.erdos_renyi_graph(n=self.number_of_households,
                                        p=self.number_of_nearest_neighbours / self.number_of_households,
                                        seed=self.seed)
        elif self.network == 'barabasi_albert':
            return nx.barabasi_albert_graph(n=self.number_of_households,
                                            m=self.number_of_edges,
                                            seed=self.seed)
        elif self.network == 'watts_strogatz':
            return nx.watts_strogatz_graph(n=self.number_of_households,
                                           k=self.number_of_nearest_neighbours,
                                           p=self.probability_of_network_connection,
                                           seed=self.seed)
        elif self.network == 'no_network':
            G = nx.Graph()
            G.add_nodes_from(range(self.number_of_households))
            return G
        else:
            raise ValueError(f"Unknown network type: '{self.network}'. "
                             f"Currently implemented network types are: "
                             f"'erdos_renyi', 'barabasi_albert', 'watts_strogatz', and 'no_network'")

    def initialize_maps(self, flood_map_choice):
        """
        Initialize and set up the flood map related data based on the provided flood map choice.
        """
        # Define paths to flood maps
        flood_map_paths = {
            'harvey': r'input_data/floodmaps/Harvey_depth_meters.tif',
            '100yr': r'input_data/floodmaps/100yr_storm_depth_meters.tif',
            '500yr': r'input_data/floodmaps/500yr_storm_depth_meters.tif'  # Example path for 500yr flood map
        }

        # Throw a ValueError if the flood map choice is not in the dictionary
        if flood_map_choice not in flood_map_paths.keys():
            raise ValueError(f"Unknown flood map choice: '{flood_map_choice}'. "
                             f"Currently implemented choices are: {list(flood_map_paths.keys())}")

        # Choose the appropriate flood map based on the input choice
        flood_map_path = flood_map_paths[flood_map_choice]

        # Loading and setting up the flood map
        self.flood_map = rs.open(flood_map_path)
        self.band_flood_img, self.bound_left, self.bound_right, self.bound_top, self.bound_bottom = get_flood_map_data(
            self.flood_map)

    def count_households_in_state(self, state):
        """
        Return the total number of households in a given state.

        Parameters:
        state (int): The state number to count.

        Returns:
        int: The number of households in the specified state.
        """
        return sum(1 for agent in self.schedule.agents if isinstance(agent, Households) and agent.state == state)

    def total_adapted_households(self):
        """
        Return the total number of households that have adapted, considering all adaptation states (1 to 4).
        """
        # Sum the number of households in states 1 to 4
        adapted_count = sum(self.count_households_in_state(state) for state in range(1, 5))
        return adapted_count

    def plot_model_domain_with_agents(self):
        fig, ax = plt.subplots()
        # Plot the model domain
        map_domain_gdf.plot(ax=ax, color='lightgrey')
        # Plot the floodplain
        floodplain_gdf.plot(ax=ax, color='lightblue', edgecolor='k', alpha=0.5)

        # Define a color map for different states
        state_colors = {0: 'red', 1: 'orange', 2: 'yellow', 3: 'green', 4: 'blue'}

        # Collect agent locations and states
        for agent in self.schedule.agents:
            agent_color = state_colors.get(agent.state, 'grey')  # Default to grey if state is not in the dictionary
            ax.scatter(agent.location.x, agent.location.y, color=agent_color, s=10,
                       label=agent_color.capitalize() if agent_color not in ax.legend_.legendHandles else "")
            ax.annotate(str(agent.unique_id), (agent.location.x, agent.location.y), textcoords="offset points",
                        xytext=(0, 1), ha='center', fontsize=9)

        # Create a legend with unique entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title="Agent States")

        # Customize plot with titles and labels
        plt.title(f'Model Domain with Agents at Step {self.schedule.steps}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()

    def information_policy_wave(self):
        """
        Function to implement the information policy wave based on the model's policy type and radius.
        There are two types: 'NL-Alert' and 'Knocking'.

        Parameters:
        model: The model instance.
        """
        policy_type = self.information_policy_type
        radius = self.information_policy_radius

        if policy_type == 'NL-Alert':
            # Randomly select 20% of households
            selected_households = random.sample(self.schedule.agents, int(0.2 * len(self.schedule.agents)))
            for agent in selected_households:
                agent.reached = True

        elif policy_type == 'Knocking':
            if radius is None:
                raise ValueError("Radius must be provided for 'Knocking' information policy.")

            # Randomly select one household
            initial_household = random.choice(self.schedule.agents)
            # Inform households within the specified radius
            for agent in self.schedule.agents:
                if self.grid.get_distance(agent.pos, initial_household.pos) <= radius:
                    agent.reached = True

    def step(self):
        """
        introducing a shock:
        at time step 15, there will be a global flooding.
        This will result in actual flood depth. we assume initially it is a random number
        between 0.5 and 1.2 of the estimated flood depth. (FOR FUTURE IMPLEMENTATION MAYBE: In your model, you can replace this
        with a more sound procedure (e.g., you can devide the floop map into zones and
        assume local flooding instead of global flooding). The actual flood depth can be
        estimated differently)
        """
        # Applying information policy wave
        if self.schedule.steps == 1:  # or whatever condition you choose
            self.information_policy_wave()

        if self.schedule.steps == 15:
            for agent in self.schedule.agents:
                if isinstance(agent, Households):
                    # Calculate flood damage depending on adaptation state
                    agent.calculate_flood_damage()

        # Collect data and advance the model by one step
        self.datacollector.collect(self)
        self.schedule.step()


if __name__ == "__main__":
    adaptation_model = AdaptationModel()
    for i in range(50):  # Run for 50 steps
        adaptation_model.step()