import random

def generate_random_routes(filename="random_routes.rou.xml", num_vehicles=100, depart_interval=1.0):
    edges_in = ['-north', '-east', '-south', '-west']
    edges_out = ['north', 'east', 'south', 'west']

    with open(filename, "w") as f:
        f.write("<routes>\n")
        f.write('  <vType id="car" accel="1.0" decel="4.5" maxSpeed="25" length="5"/>\n')

        for i in range(num_vehicles):
            while True:
                from_edge = random.choice(edges_in)
                to_edge = random.choice(edges_out)
                # prevent U-turns
                if from_edge[1:] != to_edge:
                    break

            depart_time = i * depart_interval
            f.write(f'  <route id="r{i}" edges="{from_edge} {to_edge}"/>\n')
            f.write(f'  <vehicle id="veh{i}" type="car" route="r{i}" depart="{depart_time:.1f}"/>\n')

        f.write("</routes>\n")

    print(f"Generated {num_vehicles} randomized vehicles in {filename}")
