import geopandas as gpd
import pandas as pd

# Load the datasets
roads = gpd.read_file('./output/top200_CoolCorridors.geojson')
ntas = gpd.read_file('./output/CC_NTA.geojson')

# Perform the spatial join: each row corresponds to an intersection between an NTA and a road.
joined = gpd.sjoin(ntas, roads, how='left', predicate='intersects')

# Select only the desired columns and remove duplicate rows (if any)
result = joined[['ntaname', 'Street']].drop_duplicates()

# Export the result to CSV
result.to_csv('./output/nta_roads_intersections.csv', index=False)

print("CSV exported to ./output/nta_roads_intersections.csv")