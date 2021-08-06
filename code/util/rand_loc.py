import random
import json

bounds = [-10, 10]
num_pts = 50
output_file = "loc.json"

locs = []
for _ in range(num_pts):
    rand_x = random.uniform(bounds[0], bounds[1])
    rand_y = random.uniform(bounds[0], bounds[1])
    locs.append([rand_x, rand_y])


with open(output_file, "w") as f:
    json.dump(locs, f)