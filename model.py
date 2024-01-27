# Importing necessary libraries
import networkx as nx
import math
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
                 number_of_households=0,  # number of household agents
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

        # Print statements for verifying initial parameters
        print(f"Initializing model with {number_of_households} households")
        print(f"Flood map choice: {flood_map_choice}")
        print(f"Network type: {network}")
        print(f"Information policy: {information_policy_type}, Radius: {information_policy_radius}")

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
        #batchrun codes
        self.running = True




        # generating the graph according to the network used and the network parameters specified
        self.G = self.initialize_network()
        print(f"Network initialized with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
        # create grid out of network graph
        self.grid = NetworkGrid(self.G)
        print("Grid initialized")
        # Initialize maps
        self.initialize_maps(flood_map_choice)
        print("Maps initialized")
        # set schedule for agents
        self.schedule = RandomActivation(self)  # Schedule for activating agents

        # create households through initiating a household on each node of the network graph
        print("Creating and placing households on the grid:")
        for i, node in enumerate(self.G.nodes()):
            household = Households(unique_id=i, model=self)
            self.schedule.add(household)
            self.grid.place_agent(agent=household, node_id=node)
            # Print the position of each household
            print(f"Household {i} placed at node {node}")

        print(f"{self.schedule.get_agent_count()} households created and placed on the grid")


        # Data collection setup to collect data
        model_metrics = {
            "total_HH_state_0": lambda m: sum(1 for agent in m.schedule.agents if agent.state == 0),
            "total_HH_state_1": lambda m: sum(1 for agent in m.schedule.agents if agent.state == 1),
            "total_HH_state_2": lambda m: sum(1 for agent in m.schedule.agents if agent.state == 2),
            "total_HH_state_3": lambda m: sum(1 for agent in m.schedule.agents if agent.state == 3),
            "total_HH_state_4": lambda m: sum(1 for agent in m.schedule.agents if agent.state == 4),
            "average_actual_flood_depth": lambda m: sum(agent.flood_depth_actual for agent in m.schedule.agents) / len(
                m.schedule.agents),
            "average_actual_flood_damage": lambda m: sum(
                agent.flood_damage_actual for agent in m.schedule.agents) / len(m.schedule.agents),
            "proportion_reached_by_info_policy": lambda m: sum(1 for agent in m.schedule.agents if agent.reached) / len(
                m.schedule.agents),
            "proportion_in_floodplain": lambda m: sum(1 for agent in m.schedule.agents if agent.in_floodplain) / len(
                m.schedule.agents),
            "average_init_flood_damage": lambda m: sum(
                agent.initial_damage for agent in m.schedule.agents) / len(m.schedule.agents),
            # Add or modify other metrics as needed
        }

        agent_metrics = {
            "FloodDepthActual": "flood_depth_actual",
            "FloodDamageActual": "flood_damage_actual",
            "InitFloodDmg": "initial_damage",
            "State": "state",
            "InFloodplain": "in_floodplain",
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

        # Define a color map for different states (esempio generico)
        state_colors = {0: 'red', 1: 'orange', 2: 'yellow', 3: 'green', 4: 'blue'}

        # Inizializzazione di un insieme per tenere traccia delle etichette già aggiunte
        added_labels = set()

        for agent in self.schedule.agents:
            agent_state = agent.state  # Assumi che ogni agente abbia una proprietà 'state'
            agent_color = state_colors.get(agent_state, 'grey')  # Colore di default se lo stato non è nel dizionario

            if agent_color not in added_labels:
                ax.scatter(agent.location.x, agent.location.y, color=agent_color, s=10, label=agent_color)
                added_labels.add(agent_color)
            else:
                ax.scatter(agent.location.x, agent.location.y, color=agent_color, s=10)

            # Opzionale: annotare gli agenti con il loro ID univoco
            ax.annotate(str(agent.unique_id), (agent.location.x, agent.location.y), textcoords="offset points",
                        xytext=(0, 1), ha='center', fontsize=9)

        # Create a legend with unique entries
        ax.legend(title="Agent States")

        # Customize plot with titles and labels
        plt.title(f'Model Domain with Agents at Step {self.schedule.steps}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()

    def information_policy_wave(self):
        """
        Executes an information policy wave based on the specified policy type and radius.
        The policy can either be 'NL-Alert' or 'Knocking', influencing the households accordingly.
        """
        print("Executing information policy wave...")  # Debug statement
        policy_type = self.information_policy_type
        radius = self.information_policy_radius
        print(f"Policy Type: {policy_type}, Radius: {radius}")  # Debug statement

        if policy_type == 'NL-Alert':
            # Randomly select 20% of households
            selected_households = random.sample(self.schedule.agents, int(0.2 * len(self.schedule.agents)))
            for agent in selected_households:
                agent.reached = True
            print(f"NL-Alert policy reached {len(selected_households)} agents.")  # Debug statement

        elif policy_type == 'Knocking':
            if radius is None:
                raise ValueError("Radius must be provided for 'Knocking' information policy.")

            # Randomly select one household
            initial_household = random.choice(self.schedule.agents)
            print(f"Knocking policy initial household ID: {initial_household.unique_id}")  # Debug statement

            # Get neighbors of the initial household using get_neighbors
            neighbors = self.get_spatial_neighbors(initial_household, radius)
            for neighbor in neighbors:
                neighbor.reached = True
            print(f"Knocking policy reached {len(neighbors)} agents.")

            # Inform households within the specified radius
            reached_agents_count = 0  # Debug variable
            for neighbor in neighbors:
                neighbor.reached = True
                reached_agents_count += 1  # Debug increment
            print(f"Knocking policy reached {reached_agents_count} agents within radius.")  # Debug statement

    def get_spatial_neighbors(self, agent, radius):
        """
        Find agents within a specified radius of a given agent using spatial coordinates.

        Parameters:
        agent (Households): The agent from which to measure the distance.
        radius (float): The radius within which to search for neighbors.

        Returns:
        list: A list of agents within the specified radius.
        """
        neighbors = []
        for other_agent in self.schedule.agents:
            if other_agent is agent:
                continue  # Skip the same agent

            # Calculate the Euclidean distance between agents
            distance = math.sqrt((agent.location.x - other_agent.location.x) ** 2 +
                                 (agent.location.y - other_agent.location.y) ** 2)
            if distance <= radius:
                neighbors.append(other_agent)

        return neighbors

    def step(self):
        """
        Executing the model steps. Introduces a shock with global flooding at step 14.
        """
        # Applying information policy wave
        if self.schedule.steps == 1:  # or other specific conditions
            self.information_policy_wave()

        # At step 14, calculate the flood damage and depth for each household
        if self.schedule.steps == 14:
            total_damage = 0
            initial_damage = 0
            total_depth = 0
            valid_depth_count = 0  # Counter for valid (non-negative) depths

            for agent in self.schedule.agents:
                if isinstance(agent, Households):
                    agent.calculate_flood_damage()
                    total_damage += agent.flood_damage_actual
                    initial_damage += agent.initial_damage
                    # Include only non-negative depths in the average calculation
                    if agent.flood_depth_actual >= 0:
                        total_depth += agent.flood_depth_actual
                        valid_depth_count += 1

            # Calculate and print the average flood depth and damage
            # Check to avoid division by zero
            average_depth = (total_depth / valid_depth_count) if valid_depth_count > 0 else 0
            average_damage = total_damage / self.number_of_households
            average_initial_damage = initial_damage / self.number_of_households
            print(f"Average Flood Depth at Step 14: {average_depth}")
            print(f"Average Flood Damage at Step 14: {average_damage}")
            print(f"Average Initial Flood Damage: {average_initial_damage}")

        # Proceed with regular model operations
        self.datacollector.collect(self)
        self.schedule.step()