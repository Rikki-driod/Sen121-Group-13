import random
from mesa import Agent
from shapely.geometry import Point
from functions import generate_random_location_within_map_domain, floodplain_multipolygon, get_flood_depth, calculate_basic_flood_damage

class Households(Agent):
    """
    Represents a household in the model. Each household has a state indicating its adaptation level against flooding.
    States range from 0 (no adaptation) to 4 (maximum adaptation).
    Households can influence their neighbors to increase their adaptation level.
    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.state = 0  # Initial adaptation state (0 = no adaptation)
        self.reached = False  # Tracks if the household has been reached by the information policy

        # Set the household's location and check if it's within a floodplain
        loc_x, loc_y = generate_random_location_within_map_domain()
        self.location = Point(loc_x, loc_y)
        self.in_floodplain = floodplain_multipolygon.contains(self.location)
        #print(f"Household {self.unique_id} placed at coordinates ({loc_x}, {loc_y})")
        # Initialize actual flood depth and damage
        self.flood_depth_actual = 0  # Actual flood depth (updated during a flood event)
        self.flood_damage_actual = 0  # Actual flood damage (calculated based on flood depth)

    def step(self):
        """
        Step function called at every model step. It handles the adaptation state changes and neighbor influence.
        """

        # Check and change state if reached by information policy
        if self.reached and self.state == 0:
            self.change_state(1)

        # Influence neighbors if the household is in state 1 or higher
        if self.state >= 1:
            self.influence_neighbors()

    def influence_neighbors(self):
        """
        Tries to influence each neighbor to increase their adaptation state. Each neighbor has a 50% chance of being influenced.
        """
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        for neighbor in neighbors:
            if random.random() < 0.3:  # 50% chance to influence each neighbor
                neighbor.change_state(min(4, neighbor.state + 1))  # Increase neighbor's state by 1, up to max 4

    def change_state(self, new_state):
        """
        Changes the state of the household. Resets the influence flag to allow influencing again in the new state.
        """
        self.state = new_state
        self.reached = False  # Reset the reached flag for the next round of information policy

    def calculate_flood_damage(self):

        # Calculate the actual flood depth
        self.flood_depth_actual = get_flood_depth(self.model.flood_map, self.location, self.model.band_flood_img)


        # Calculate initial damage based on the flood depth
        initial_damage = calculate_basic_flood_damage(self.flood_depth_actual)


        # Reduction factors for each state (from 0 to 4)
        damage_reduction_factor = [1, 0.75, 0.5, 0.3, 0.2]  # Adjust these values as needed

        # Apply the reduction factor based on the household's state
        self.flood_damage_actual = initial_damage * damage_reduction_factor[self.state]

        print(f"Household {self.unique_id} - Flood Depth: {self.flood_depth_actual}, "
              f"Initial Damage: {initial_damage}, State: {self.state}, "
              f"Final Damage: {self.flood_damage_actual}")