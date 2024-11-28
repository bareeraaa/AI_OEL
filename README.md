# AI_OEL
Traveling Salesman Problem (TSP) Using Genetic Algorithm
This project solves the Traveling Salesman Problem (TSP) for cities in Pakistan using a Genetic Algorithm (GA). The solution identifies an optimal route that minimizes the total distance required to visit all cities and return to the starting city.

Features
Cities of Pakistan: Includes 10 major cities with real-world latitude and longitude coordinates.
Haversine Formula: Calculates the great-circle distance between two cities in kilometers.
Genetic Algorithm:
Population Initialization with random permutations.
Fitness calculation based on the total route distance.
Selection using a roulette wheel mechanism.
Crossover and mutation for generating new populations.
Visualization:
Displays the optimal route on a map using GeoPandas and Matplotlib.
Customizable map and route colors.
Getting Started
Prerequisites
Ensure the following Python libraries are installed:

numpy
matplotlib
shapely
geopandas
You can install them with:

pip install numpy matplotlib shapely geopandas
Download the shapefile for world boundaries and update the shapefile_path in the code:

shapefile_path = "E:/ne_10m_admin_0_countries.shp"
You can download the shapefile here.

