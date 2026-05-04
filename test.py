from Environment import generate_unique_coords
import numpy as np

device_coord = generate_unique_coords(N = 10)

UAVs_init_coord = np.array([])

risky_region = [
    [3.0, 2.0, 6.0, 6.0],
]

L_map = np.array([0.0, 0.0], dtype=np.float32)
H_map = np.array([10.0, 10.0], dtype=np.float32)

def is_coord_risky(coord):
    x, y = coord

    for region in risky_region:
        xmin, ymin, xmax, ymax = region

        if xmin <= x <= xmax and ymin <= y <= ymax:
            return True

    return False


for _ in range(10):
    UAV_coord = np.random.uniform(
        low=L_map,
        high=H_map,
        size=2
    ).astype(np.float32)

    while is_coord_risky(UAV_coord):
        UAV_coord = np.random.uniform(
            low=L_map,
            high=H_map,
            size=2
        ).astype(np.float32)

    UAVs_init_coord = np.append(UAVs_init_coord, UAV_coord)

print(UAVs_init_coord)