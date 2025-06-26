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

    # Calculate roughly balanced distribution with variation
    num_routes = len(unique_routes)
    base_vehicles_per_route = num_vehicles // num_routes
    
    # Create route assignments with some randomness around the base amount
    route_assignments = []
    total_assigned = 0
    route_counts = []
    
    for i in range(num_routes - 1):  # Handle all routes except the last one
        # Add some random variation (±50% of base amount, but at least ±3)
        variation_range = max(3, int(base_vehicles_per_route * 0.5))
        vehicles_for_this_route = base_vehicles_per_route + random.randint(-variation_range, variation_range)
        
        # Ensure we don't go negative or assign too many vehicles
        vehicles_for_this_route = max(1, min(vehicles_for_this_route, num_vehicles - total_assigned - (num_routes - i - 1)))
        
        route_assignments.extend([i] * vehicles_for_this_route)
        route_counts.append(vehicles_for_this_route)
        total_assigned += vehicles_for_this_route
    
    # Assign remaining vehicles to the last route
    remaining_vehicles = num_vehicles - total_assigned
    route_assignments.extend([num_routes - 1] * remaining_vehicles)
    route_counts.append(remaining_vehicles)
    
    # Shuffle the route assignments to randomize vehicle order
    random.shuffle(route_assignments)

    with open(filename, "w") as f:
        f.write("<routes>\n")

        # Define vehicle types in XML
        for vtype, params in vehicle_types.items():
            f.write(f'  <vType id="{vtype}" accel="{params["accel"]}" decel="{params["decel"]}" '
                    f'maxSpeed="{params["maxSpeed"]}" length="{params["length"]}"/>\n')

        # Define all possible route IDs
        for idx, (from_edge, to_edge) in enumerate(unique_routes):
            f.write(f'  <route id="r{idx}" edges="{from_edge} {to_edge}"/>\n')

        # Generate vehicles with balanced route distribution
        current_depart_time = 0.0
        for i in range(num_vehicles):
            route_id = f"r{route_assignments[i]}"
            vtype = random.choice(weighted_vehicle_list)
            f.write(f'  <vehicle id="veh{i}" type="{vtype}" route="{route_id}" depart="{current_depart_time:.1f}"/>\n')
            current_depart_time += random.uniform(0, 5)

        f.write("</routes>\n")

    # Print distribution summary
    print(f"Generated {num_vehicles} vehicles distributed across {num_routes} routes:")
    for i, (from_edge, to_edge) in enumerate(unique_routes):
        print(f"  Route r{i} ({from_edge} -> {to_edge}): {route_counts[i]} vehicles")
    print(f"Saved to {filename}")
