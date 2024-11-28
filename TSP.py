import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from shapely.geometry import LineString
import geopandas as gpd
from math import radians, sin, cos, sqrt, atan2

# Define cities and coordinates for cities in Pakistan
cities_names = ["Karachi", "Lahore", "Islamabad", "Peshawar", "Quetta", 
                "Faisalabad", "Multan", "Hyderabad", "Rawalpindi", "Gujranwala"]
x = [67.0011, 74.3587, 73.0551, 71.5249, 66.9750, 74.1955, 71.5249, 68.3739, 73.0672, 74.1551]
y = [24.8615, 31.5204, 33.6844, 34.0151, 30.1798, 30.1575, 30.1575, 25.3925, 33.5950, 32.0462]
city_coords = dict(zip(cities_names, zip(x, y)))

# Initialization
def initial_population(cities_list, n_population=250):
    """Generate initial population."""
    population = []
    possible_perms = list(permutations(cities_list[1:]))  # Start with fixed "Karachi"
    random_ids = random.sample(range(len(possible_perms)), n_population)
    for i in random_ids:
        individual = ["Karachi"] + list(possible_perms[i]) + ["Karachi"]
        population.append(individual)
    return population


# Haversine formula for distance in kilometers
def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points in kilometers."""
    R = 6371  # Radius of Earth in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Fitness functions
def dist_two_cities(city_1, city_2):
    """Calculate distance between two cities in kilometers."""
    lat1, lon1 = city_coords[city_1]
    lat2, lon2 = city_coords[city_2]    
    return haversine(lat1, lon1, lat2, lon2)

def total_dist_individual(individual):
    """Calculate total distance for an individual."""
    return sum(dist_two_cities(individual[i], individual[i+1]) for i in range(len(individual) - 1))

def fitness_prob(population):
    """Calculate fitness probability for the population."""
    distances = np.array([total_dist_individual(ind) for ind in population])
    fitness = distances.max() - distances
    total_fitness = fitness.sum()
    if total_fitness == 0:
        return np.ones(len(fitness)) / len(fitness)  # Uniform probabilities if all fitness are zero
    return fitness / total_fitness


# Selection
def roulette_wheel(population, fitness_probs):
    """Select individual based on fitness probabilities."""
    cumsum_probs = np.cumsum(fitness_probs)
    random_value = np.random.rand()
    return population[np.searchsorted(cumsum_probs, random_value)]

# Crossover and Mutation
def crossover(parent_1, parent_2):
    """Perform crossover between two parents."""
    cut = random.randint(1, len(parent_1) - 2)
    child = ["Karachi"] + parent_1[1:cut] + [city for city in parent_2[1:] if city not in parent_1[1:cut]] + ["Karachi"]
    return child

def mutation(offspring):
    """Mutate offspring by swapping two cities."""
    idx1, idx2 = random.sample(range(1, len(offspring) - 1), 2)
    offspring[idx1], offspring[idx2] = offspring[idx2], offspring[idx1]
    return offspring

# Genetic Algorithm
def run_ga(cities_names, n_population, n_generations, crossover_rate, mutation_rate):
    """Run the Genetic Algorithm."""
    population = initial_population(cities_names, n_population)

    for _ in range(n_generations):
        fitness_probs = fitness_prob(population)
        next_gen = []
        for _ in range(int(crossover_rate * n_population // 2)):
            parent_1 = roulette_wheel(population, fitness_probs)
            parent_2 = roulette_wheel(population, fitness_probs)
            child_1 = crossover(parent_1, parent_2)
            child_2 = crossover(parent_2, parent_1)
            if random.random() < mutation_rate:
                child_1 = mutation(child_1)
            if random.random() < mutation_rate:
                child_2 = mutation(child_2)
            next_gen.extend([child_1, child_2])

        population = sorted(population + next_gen, key=total_dist_individual)[:n_population]

    return population

# Visualization
def plot_route(route, total_distance, generations, population_size, crossover_rate, mutation_rate, map_color='lightgray', route_color='red'):
    """Plot the best route on a map using GeoPandas."""
    shapefile_path = "E:/ne_10m_admin_0_countries.shp"  # Update with your path
    world = gpd.read_file(shapefile_path)
    pakistan = world[world['ADMIN'] == 'Pakistan']

    # Create GeoDataFrame for the route
    route_coords = [(city_coords[city][0], city_coords[city][1]) for city in route]
    route_line = LineString(route_coords)
    gdf_route = gpd.GeoDataFrame({'geometry': [route_line], 'distance': [total_distance]})

    # Plot the map and route
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Change map color (landmass color)
    pakistan.plot(ax=ax, color=map_color, edgecolor='black', alpha=0.7)
    
    # Change route color (dotted line)
    gdf_route.plot(ax=ax, linestyle='--', color=route_color, linewidth=2, label='Optimal Route (Dotted Line)')

    # Annotate cities
    for idx, city in enumerate(route):
        x, y = city_coords[city]
        ax.annotate(f"{idx + 1} - {city}" if city != "Karachi" else f"1 - Karachi", 
                    (x, y), fontsize=9, ha='right')

    ax.set_title(f"TSP Optimal Route\nDistance: {round(total_distance, 3)} km\n"
                 f"Generations: {generations}, Population: {population_size}, "
                 f"Crossover: {crossover_rate}, Mutation: {mutation_rate}")
    plt.legend()
    plt.show()


# Run the GA
n_population = 250
n_generations = 100
crossover_rate = 0.8
mutation_rate = 0.1

population = run_ga(cities_names, n_population, n_generations, crossover_rate, mutation_rate)

# Find the shortest route
distances = [total_dist_individual(ind) for ind in population]
shortest_path = population[np.argmin(distances)]
minimum_distance = min(distances)

# Plot the best route with custom colors
plot_route(shortest_path, minimum_distance, n_generations, n_population, crossover_rate, mutation_rate, map_color='lightblue', route_color='green')
