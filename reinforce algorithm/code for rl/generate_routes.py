# import random

# def generate_random_routes(filename="random_routes.rou.xml", num_vehicles=100, depart_interval=1.0):
#     edges_in = ['-north', '-east', '-south', '-west']
#     edges_out = ['north', 'east', 'south', 'west']

#     with open(filename, "w") as f:
#         f.write("<routes>\n")
#         f.write('  <vType id="car" accel="1.0" decel="4.5" maxSpeed="25" length="5"/>\n')

#         for i in range(num_vehicles):
#             while True:
#                 from_edge = random.choice(edges_in)
#                 to_edge = random.choice(edges_out)
#                 # prevent U-turns
#                 if from_edge[1:] != to_edge:
#                     break

#             depart_time = i * depart_interval
#             f.write(f'  <route id="r{i}" edges="{from_edge} {to_edge}"/>\n')
#             f.write(f'  <vehicle id="veh{i}" type="car" route="r{i}" depart="{depart_time:.1f}"/>\n')

#         f.write("</routes>\n")

#     print(f"Generated {num_vehicles} randomized vehicles in {filename}")

import random

def generate_random_routes(filename="random_routes.rou.xml", num_vehicles=100, type_ratios=None):
    edges_in = ['-north', '-east', '-south', '-west']
    edges_out = ['north', 'east', 'south', 'west']

    vehicle_types = {
        'car': {'accel': 1.0, 'decel': 4.5, 'length': 5, 'maxSpeed': 25},
        'truck': {'accel': 0.5, 'decel': 3.0, 'length': 12, 'maxSpeed': 18},
        'bus': {'accel': 0.8, 'decel': 3.5, 'length': 10, 'maxSpeed': 20},
        'scooter': {'accel': 1.5, 'decel': 4.0, 'length': 2, 'maxSpeed': 30}
    }

    if type_ratios is None:
        type_ratios = {'car': 0.5, 'truck': 0.1, 'bus': 0.2, 'scooter': 0.2}

    # Normalize ratios
    total_ratio = sum(type_ratios.values())
    for key in type_ratios:
        type_ratios[key] /= total_ratio

    # Create weighted list of vehicle types
    weighted_vehicle_list = []
    for vtype, ratio in type_ratios.items():
        weighted_vehicle_list.extend([vtype] * int(ratio * 1000))  # resolution of 1000

    # Predefine unique (from, to) routes
    unique_routes = []
    for from_edge in edges_in:
        for to_edge in edges_out:
            if from_edge[1:] != to_edge:
                unique_routes.append((from_edge, to_edge))

    with open(filename, "w") as f:
        f.write("<routes>\n")

        # Define vehicle types in XML
        for vtype, params in vehicle_types.items():
            f.write(f'  <vType id="{vtype}" accel="{params["accel"]}" decel="{params["decel"]}" '
                    f'maxSpeed="{params["maxSpeed"]}" length="{params["length"]}"/>\n')

        # Define all possible route IDs
        for idx, (from_edge, to_edge) in enumerate(unique_routes):
            f.write(f'  <route id="r{idx}" edges="{from_edge} {to_edge}"/>\n')

        # Generate vehicles
        current_depart_time = 0.0
        for i in range(num_vehicles):
            route_id = f"r{random.randint(0, len(unique_routes) - 1)}"
            vtype = random.choice(weighted_vehicle_list)
            f.write(f'  <vehicle id="veh{i}" type="{vtype}" route="{route_id}" depart="{current_depart_time:.1f}"/>\n')
            current_depart_time += random.uniform(0, 5)

        f.write("</routes>\n")

    print(f"Generated {num_vehicles} vehicles using type ratios in {filename}")


