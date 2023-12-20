import random
from mesa import Agent
from shapely.geometry import Point
from shapely import contains_xy
from functions import generate_random_location_within_map_domain, calculate_basic_flood_damage, floodplain_multipolygon

class Households(Agent):
    """
    Represents a household in the model. Each household has a state indicating its adaptation level against flooding.
    States range from 0 (no adaptation) to 4 (maximum adaptation).
    Households can influence their neighbors to increase their adaptation level.
    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.level_of_knowledge = random.uniform(0, 1)  # Randomly assigned knowledge level
        self.state = 0  # Initial adaptation state (0 = no adaptation)
        self.reached = False  # Tracks if the household has been reached by the information policy
        self.influenced_this_state = False  # Tracks if the household has influenced neighbors in the current state

        # Set the household's location and check if it's within a floodplain
        loc_x, loc_y = generate_random_location_within_map_domain()
        self.location = Point(loc_x, loc_y)
        self.in_floodplain = contains_xy(geom=floodplain_multipolygon, x=self.location.x, y=self.location.y)

        # Initialize actual flood depth and damage
        self.flood_depth_actual = 0  # Actual flood depth (updated during a flood event)
        self.flood_damage_actual = 0  # Actual flood damage (calculated based on flood depth)

    def step(self):
        """
        Step function called at every model step. It handles the adaptation state changes and neighbor influence.
        """
        # If reached by information policy and in state 0, move to state 1
        if self.reached and self.state == 0:
            self.change_state(1)

        # Influence neighbors if the household is in state 1 or higher and hasn't influenced in the current state
        if self.state >= 1 and not self.influenced_this_state:
            self.influence_neighbors()

    def influence_neighbors(self):
        """
        Attempts to influence each neighbor to increase their adaptation state. Each neighbor has a 50% chance to be influenced.
        """
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        for neighbor in neighbors:
            if random.random() < 0.5:  # 50% chance to influence each neighbor
                neighbor.react_to_influence(self.level_of_knowledge)

    def react_to_influence(self, influencing_knowledge):
        """
        Reacts to the influence from a neighbor. Higher knowledge level of the influencer increases the chance of state change.
        """
        if self.state < 4 and random.random() < influencing_knowledge:
            self.change_state(self.state + 1)

    def change_state(self, new_state):
        """
        Changes the state of the household. Resets the influence flag to allow influencing again in the new state.
        """
        if new_state != self.state:
            self.state = new_state
            self.influenced_this_state = False  # Reset the flag to allow influence in the new state

    def calculate_flood_damage(self):
        """
        Calculates flood damage based on the actual flood depth. The damage calculation is influenced by the adaptation state.
        """
        self.flood_depth_actual = random.uniform(0.5, 1.2) * self.flood_depth_actual
        damage_reduction_factor = [1, 0.8, 0.5, 0.2, 0.2]  # Reduction factors for each state
        self.flood_damage_actual = calculate_basic_flood_damage(self.flood_depth_actual) * damage_reduction_factor[self.state]


