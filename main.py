import random
import psutil
import pandas as pd
import datetime
import time
from memory_profiler import memory_usage
from bidding_graph import create_bidding_graph

def dataProcessing(bidding, source, target):

    # convert the start and end times to time objects
    start = []
    end = []
    source_nodes = []
    target_nodes = []

    for index, row in bidding.iterrows():

         # extract the start and end times from the string
        start_str_list = row["pickup time"].strip("[]").split("-")
        end_str_list = row["delivery time"].strip("[]").split("-")

        start_time_str = start_str_list[0]
        end_time_str = end_str_list[0]

        # use strptime() to convert the start and end times to time objects
        start_time_obj = datetime.datetime.strptime(start_time_str, "%H:%M:%S").time()
        end_time_obj = datetime.datetime.strptime(end_time_str, "%H:%M:%S").time()
        start.append(start_time_obj)
        end.append(end_time_obj)

    # add the start and end time objects to the data frmae
    bidding["pickup time"] = start
    bidding["delivery time"] = end

    # Marking bidding starting from the source point and/or ending in the target point
    for index, row in bidding.iterrows():

        if row["origin"] == source:
            source_nodes.append(True)
        else:
            source_nodes.append(False)
        if row["dest"] == target:
            target_nodes.append(True)
        else:
            target_nodes.append(False)

    bidding["source?"] = source_nodes
    bidding["target?"] = target_nodes


def initialize_population(Population_size, start_nodes):

    population = {}
    for i in range(Population_size):
        population[i] = []

    for i in range(Population_size):
        # append random bidding from the start bidding For each chromosome
        index = random.randint(0, len(start_nodes) - 1)
        population[i].append(start_nodes[index])

    # Generating a complete route from the start node
    for i in range(Population_size):

        destination = G.nodes[population[i][-1]]["destination_location"]

        # while the chromosome is not a full route
        count=0
        while destination != DESTINATION :
            count = count+1

            neighbors = list(G.neighbors(population[i][-1]))
            if len(neighbors) != 0:
                # select randomly a neighbor to be the next location in the route
                index = random.randint(0, len(neighbors) -1)

                # add sub-route to the chromosome
                population[i].append(neighbors[index])
            else:
                # remove the chromosome from the population if it has no more neighbors
                del population[i]
                population[i] = []
                index = random.randint(0, len(start_nodes) - 1)
                population[i].append(start_nodes[index])
                continue

            destination = G.nodes[population[i][-1]]["destination_location"]

    return population

def evaluate_chromosome(chromosome):
        total_cost = 0
        for node in chromosome:
            total_cost += G.nodes[node]['price']
        return total_cost


def evaluate_popualtion(chromosome):
    total_cost = 0
    for node in chromosome:
        total_cost += G.nodes[node]['price']
    return total_cost
def crossover1(parent1 , parent2):

    cut_points = []
    child1 = []
    child2 = []
    for i in range(len(parent1)):
        for j in range(len(parent2)):
            gen1 = parent1[i]
            gen2 = parent2[j]
            if G.nodes[gen1]['source_location'] == G.nodes[gen2]['source_location'] and G.nodes[gen1]['destination_location'] == G.nodes[gen2]['destination_location']:
                cut_points.append((i, j))
    if len(cut_points) > 0:
        to_cross = random.choice(cut_points)
        for i in range(0, to_cross[0]+1):
            child1.append(parent1[i])

        for j in range(to_cross[1]+1,len(parent2)):
            child1.append(parent2[j])

        for j in range(0,to_cross[1]+1):
            child2.append(parent2[j])

        for i in range(to_cross[0]+1,len(parent1)):
            child2.append(parent1[i])
        return child1, child2
    else: return parent1,parent2


def crossover3(chromosome):

    gen = random.choice(chromosome)
    node_gen = G.nodes[gen]
    to_cross = [node.ID for node in nodes_by_source[node_gen["source_location"]] if node.destination_location == node_gen["destination_location"] and node.ID != gen]
    child = chromosome.copy()

    if len(to_cross) != 0:
        #new_gen = min(to_cross, key=lambda node: G.nodes[node]['price'])
        new_gen = random.choice(to_cross)
        index = chromosome.index(gen)

        tmpChild = child[: -(len(child)-index)]

        tmpChild.append(new_gen)
        destination = G.nodes[new_gen]["destination_location"]

        # while the chromosome is not a full route
        count = 0
        while destination != DESTINATION :
            count = count + 1
            neighbors = list(G.neighbors(tmpChild[-1]))
            if len(neighbors) != 0:
                # select randomly a neighbor to be the next location in the route
                index = random.randint(0, len(neighbors) - 1)
                lastGen = neighbors[index]
                # add sub-route to the chromosome
                tmpChild.append(lastGen)
            else:

                # remove the chromosome from the population if it has no more neighbors
                tmpChild = child[: -(len(child) - index)].copy()
                tmpChild.append(new_gen)

            destination = G.nodes[tmpChild[-1]]["destination_location"]
        child = tmpChild.copy()


    return child

def crossover2(chromosome):

    gen = random.choice(chromosome)
    node_gen = G.nodes[gen]
    to_cross = [node.ID for node in nodes_by_source[node_gen["source_location"]] if node.destination_location == node_gen["destination_location"] and node.ID != gen]
    child = chromosome.copy()
    if len(to_cross) != 0:
        new_gen = min(to_cross, key=lambda node: G.nodes[node]['price'])
        #new_gen = random.choice(to_cross)
        index = chromosome.index(gen)

        child[index] = new_gen

    return child

def selection(population):

    selected_parents = []
    population_list = list(population.values())
    tournament_size = 10
    while len(selected_parents) < 2:
        # Select random individuals for the tournament
        tournament = random.sample(population_list, tournament_size)

        # Evaluate fitness for each individual in the tournament
        fitness_values = [evaluate_chromosome(chromosome) for chromosome in tournament]

        # Find the individual with the lowest fitness (minimization problem)
        best_index = fitness_values.index(min(fitness_values))

        # Select the best individual from the tournament as a parent
        selected_parents.append(tournament[best_index])

    return selected_parents[0], selected_parents[1]



def print_population(population):
    n = len(population)
    for i in range(n):
        t_cost = 0
        for node in population[i]:
             t_cost += G.nodes[node]["price"]

        print(f"chromosome {i}: bidders: {[G.nodes[node]['ID'] for node in population[i]]} path:{[G.nodes[node]['segment'] for node in population[i]]} pickuptimes:{[G.nodes[node]['pickup_time'].strftime('%H:%M:%S')  for node in population[i]]} delivery times:{[G.nodes[node]['delivery_time'].strftime('%H:%M:%S')  for node in population[i]]}  fitness={t_cost}")


def mutation(child1):
    gen = random.choice(child1)
    to_cross = [node for node in G.nodes if G.nodes[node]["source_location"] == G.nodes[gen]["source_location"] and G.nodes[node]["destination_location"] == G.nodes[gen]["destination_location"] and node != gen]
    child = child1.copy()
    if len(to_cross) != 0:
        new_gen = random.choice(to_cross)
        index = child1.index(gen)
        child[index] = new_gen
    return child


def genetic_algo(population_size, max_generations, mutation_rate):

    # Define the Genetic Algorithm parameters
    POPULATION_SIZE = population_size
    MAX_GENERATIONS = max_generations
    MUTATION_RATE = mutation_rate
    start_nodes = [node for node in G.nodes if G.nodes[node]["source_location"] == SOURCE]
    population = initialize_population(POPULATION_SIZE, start_nodes)
    print_population(population)

    for generation in range(MAX_GENERATIONS):

        parent1, parent2 = selection(population)
        child1 = crossover2(parent1)
        child2 = crossover2(parent2)
        if random.random() < MUTATION_RATE:

             child2 = mutation(child2)

        if random.random() < MUTATION_RATE:

             child2 = mutation(child2)

        #replace  the childs with the worst chromosome in order to create new population in the smae size
        if child1 not in population.values() and len(child1) > 0:

            max_sum = float('-inf')  # Initialize with a very low value
            max_key = None

            for key, value in population.items():

                current_sum = sum(G.nodes[item]['price'] for item in value)
                if current_sum > max_sum:
                    max_sum = current_sum
                    max_key = key


            population[max_key] = child1

        if child2 not in population.values() and len(child2) > 0:
            max_sum = float('-inf')
            max_key = None
            for key, value in population.items():
                current_sum = sum(G.nodes[item]['price'] for item in value)
                if current_sum > max_sum:
                    max_sum = current_sum
                    max_key = key

            population[max_key] = child2

    min = evaluate_chromosome(population[0])
    id = 0
    for i in range(1, len(population)):
        total_cost = evaluate_chromosome(population[i])
        if total_cost < min:
            id = i
            min = total_cost
    best_chromosome = population[id].copy()

    print("result:")
    print(f"chromosome: {id},{[node for node in best_chromosome]},total price = {min}")
    return min
def h3(node):
    avg_fuel = 6.9
    avg_fuel_consumption = 11
    fuel_cost  = G.nodes[node]["distance_to_target"]/avg_fuel_consumption * avg_fuel*0.5
    return fuel_cost

def h(node):
    avg_fuel=6.9
    avg_fuel_consumption  = 11
    avg_speed = 50
    fuel_cost  = G.nodes[node]["distance_to_target"]/avg_fuel_consumption * avg_fuel
    driving_time  =  G.nodes[node]["distance_to_target"]/avg_speed
    driver_cost  = driving_time * 8
    total_cost  = fuel_cost + driver_cost
    if driving_time > 1 :
            total_cost *= 0.7
    elif driving_time > 0.5 :
        total_cost *= 0.6
    else:
        total_cost *= 0.5
    return total_cost

def h2(node):
    avg_fuel=6.9
    avg_fuel_consumption = 11
    avg_speed = 50
    fuel_cost = G.nodes[node]["distance_to_target"]/avg_fuel_consumption * avg_fuel
    driving_time  =  G.nodes[node]["distance_to_target"]/avg_speed
    if driving_time > 1 :
            fuel_cost *= 0.7
    elif driving_time > 0.5 :
        fuel_cost *= 0.6
    else:
        fuel_cost *= 0.5
    return fuel_cost

def IDA_star(graph, source, target, max_price):
    # Initialize threshold to be the heuristic estimation cost from sourch to target
    threshold = h(source)
    # Initialize path with source node
    path = [source]

    f_val = 0
    time.sleep(2)
    while f_val is not float('inf'):
        f_val, temp_path = search(graph, path, 0, threshold, target)
        if f_val == "FOUND":
            price=0
            for node in temp_path:
                price += graph.nodes[node]["price"]
            if price > max_price:
                return None
            return temp_path, price
        if f_val == float('inf'):
            return None
        threshold = f_val


def search(graph, path, g_value, threshold, target_node):

    # get the last node in path
    current_node = path[-1]

    # calculate tha f value of the current node (cost until this node [include the node price] + heuristic estimation cost from this node to target)
    current_f_value = g_value+h(current_node)

    # if the node's f value exceeded threshold we return the f value
    if current_f_value > threshold:
        return current_f_value, path

    # if the current node is the target return !!!
    if current_node == target_node:
        return "FOUND", path

    # initialize minimum price to infinity, this var will save the minimum f value so far found
    min_f = float('inf')
    dead_end_flag = True
    sortedN=sorted(graph.neighbors(current_node),key=lambda n:h(n)+graph.nodes[n]['price'])
    # loop over all current node neighbors
    for succ in sortedN:
        # if this neighbor not already in path (avoid circles)
        if succ not in path:
            path.append(succ)
            successor_cost = g_value + graph.nodes[succ]['price']

            # search in the successors of the successor
            f_val_temp, path = search(graph, path, successor_cost, threshold, target_node)

            if f_val_temp == "FOUND":
                return "FOUND", path

            # if the successor's f value is smaller than the minimum f value so far found
            if f_val_temp < min_f:
                dead_end_flag=False
                min_f = f_val_temp

            # remove last node from path
            path.pop()
    if dead_end_flag == True:
        graph.nodes[current_node]['distance_to_target']=float('inf')
    return min_f, path


########################################################### START OF DIJKSTRA #########################H###############


def dijkstra_with_price(graph, source, target, max_price):
    start_time = time.time()
    global count1
    # Initialize the cost of each node to infinity, except for the source node which is 0
    cost = {node: float('inf') for node in graph.nodes}
    cost[source] = 0
    cost[target] = float('inf')
    # Initialize the previous node from the source to each node as unknown
    prev = {node: None for node in graph.nodes}

    # Create a set of unvisited nodes
    unvisited = set(graph.nodes)
    path = [target]
    # Loop until all nodes have been visited
    time.sleep(2)
    while unvisited:

        # Find the node with the smallest cost
        current_node = min(unvisited, key=lambda node: cost[node])

        # Visit the current node
        unvisited.remove(current_node)

        # Update the cost of each neighbor of the current node
        for neighbor in graph.neighbors(current_node):
            new_cost = cost[current_node] + graph.nodes[neighbor]['price']  #neighbor price it means the price a segment
            if new_cost < cost[neighbor]:
                cost[neighbor] = new_cost
                prev[neighbor] = current_node

    # If the cost of the rout exceed the maximum price return None
    if cost[target] > max_price:
        return None

    # Build the shortest path from the source to the target node
    node = target
    while prev[node] != source:
        path.append(prev[node])
        node = prev[node]
    path.append(source)
    path.reverse()

    # Return the cost of the shortest path and the path itself
    return cost[target], path
########################################################### END OF DIJKSTRA #############################################


if __name__ == '__main__':

    SOURCE = "A"
    DESTINATION = "D"
    bidding = pd.read_csv("Large320.csv")
    dataProcessing(bidding, SOURCE, DESTINATION)
    nodes, edges, G, nodes_by_source = create_bidding_graph(bidding)

    START = "start"
    TARGET = "target"

    # Run algorithms and measure runtime/memory usage
    source, target = 0, len(G) - 1
    n_iterations = 10
    attr = {genetic_algo:(20, 10, 0.1),dijkstra_with_price:(G, 'start','target',1000),IDA_star:(G,'start', 'target',1000)}

    for algo in [IDA_star,dijkstra_with_price,genetic_algo]:
        runtimes = []
        mem_usages = []
        return_values = []
        for i in range(n_iterations):
            start_time =time.time()
            mem_usage = max(memory_usage((algo,attr[algo]), interval=0.1))
            end_time = time.time()
            runtime = end_time - start_time
            cpu_percent = psutil.cpu_percent()
            runtimes.append(runtime)
            mem_usages.append(mem_usage)
        avg_runtime = sum(runtimes) / len(runtimes)
        avg_mem_usage = sum(mem_usages) / len(mem_usages)

        print( f"Algorithm {algo.__name__}: average runtime = {avg_runtime:.4f} s, average memory usage = {avg_mem_usage:.4f} MB")


