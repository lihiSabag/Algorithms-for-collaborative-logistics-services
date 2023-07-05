import random
import sys
import time
import pandas as pd
import datetime
import timeit
import networkx as nx
import time
from memory_profiler import memory_usage


# GLOBAL VARIABLE
from bidding_graph import create_bidding_graph

edges = []
nodes = []
DESTINATION = "D"

SOURCE = "S"
DESTINATION = "D"
bidding = pd.read_csv("largeDS.csv")

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

dataProcessing(bidding, SOURCE, DESTINATION)
nodes, edges, G, N = create_bidding_graph(bidding)
# Initialize the Population
def initialize_population(Population_size, start_nodes):

    print('################  Initialize Population ################')

    population = {}
    for i in range(Population_size):
        population[i] = []

    for i in range(Population_size):
        # append random bidding from the start bidding For each chromosome
        index = random.randint(0, len(start_nodes) - 1)
        population[i].append(start_nodes[index])

    # Generating a complete route from the start node
    for i in range(Population_size):
        #destination = G.nodes[population[i][-1]].destination_location
        destination = G.nodes[population[i][-1]]["destination_location"]

        # while the chromosome is not a full route
        count=0
        while destination != DESTINATION and count <20:
            count = count+1
            #neighbors = population[i][-1].neighbors
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
            #destination = population[i][-1].destination_location
            destination = G.nodes[population[i][-1]]["destination_location"]

    return population

def evaluate_chromosome(chromosome):
        total_cost = 0
        for node in chromosome:
            #total_cost += node.price
            total_cost += G.nodes[node]['price']
        return total_cost

def crossover(chromosome):

    gen = random.choice(chromosome)
    #to_cross = [node for node in nodes if node.source_location == gen.source_location and node.destination_location == gen.destination_location and node.ID != gen.ID]
    to_cross = [node for node in G.nodes if G.nodes[node]["source_location"] == G.nodes[gen]["source_location"] and G.nodes[node]["destination_location"] == G.nodes[gen]["destination_location"] and node != gen]
    child = chromosome.copy()

    if len(to_cross) != 0:
        new_gen = random.choice(to_cross)
        index = chromosome.index(gen)
        child[index] = new_gen
    return child

def new_crossover(parent1 , parent2):
    print("new cross")
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
        print(f"tooo_cross={to_cross}")
        for i in range(0, to_cross[0]+1):
            child1.append(parent1[i])

        for j in range(to_cross[1]+1,len(parent2)):
            child1.append(parent2[j])

        for j in range(0,to_cross[1]+1):
            child2.append(parent2[j])

        for i in range(to_cross[0]+1,len(parent1)):
            child2.append(parent1[i])
    return child1,child2

import random


def selection(population):
    # Compute fitness values
    population_list = list(population.values())
    fitness_values = [evaluate_chromosome(chromosome) for chromosome in population_list]

    # Compute selection probabilities
    inverse_fitness_values = [1 / fitness_value for fitness_value in fitness_values]
    total_inverse_fitness = sum(inverse_fitness_values)
    selection_probabilities = [inverse_fitness_value / total_inverse_fitness for inverse_fitness_value in
                               inverse_fitness_values]

    # Rank population by selection probabilities
    ranked_population = [chromosome for _, chromosome in sorted(zip(selection_probabilities, population_list), reverse=True)]

    # Select parent chromosomes using inverse proportional selection
    cumulative_probabilities = [sum(selection_probabilities[:i + 1]) for i in range(len(selection_probabilities))]
    parent1 = None
    parent2 = None
    while parent1 is None:
        rand_num = random.random()
        for i, cumulative_probability in enumerate(cumulative_probabilities):
            if cumulative_probability >= rand_num:
                parent1 = ranked_population[i]
                break

    while parent2 is None:
        rand_num = random.random()
        for i, cumulative_probability in enumerate(cumulative_probabilities):
            if cumulative_probability >= rand_num:
                parent2 = ranked_population[i]
                break

    return parent1, parent2


def select_2_parents(population):

    #Random selection of a chromosome from the population as parent
    parent1 = random.choice(population).copy()
    #print(f"parent 1 = {[node.ID for node in parent1]}")
    #print(f"parent 1 = {[G.nodes[node]['ID'] for node in parent1]}")
    #Selection of the chromosome with the minimum total cost as parent
    min = evaluate_chromosome(population[0])
    id = 0
    for i in range(1, len(population)):
        total_cost = evaluate_chromosome(population[i])
        if total_cost < min:
            id = i
            min = total_cost
    parent2 = population[id].copy()
   # print(f"parent 2 = {[ node.ID for node in parent2]}")
   # print(f"parent 2 = {[G.nodes[node]['ID'] for node in parent2]}")
    return parent1, parent2

def print_population(population):
    n = len(population)
    print(f"N={n}")
    totalcost_list = []
    for i in range(n):
        t_cost = 0

        for node in population[i]:
             #t_cost += node.price

             t_cost += G.nodes[node]["price"]

        #print(f"population {i}: bidders: {[node.ID for node in population[i]]} path:{[node.segment for node in population[i]]} fitness={t_cost}")
        print(f"population {i}: bidders: {[G.nodes[node]['ID'] for node in population[i]]} path:{[G.nodes[node]['segment'] for node in population[i]]} fitness={t_cost}")
        #totalcost_list.append(t_cost)
    #return min(totalcost_list)



def genetic_algo(population_size, max_generations, mutation_rate):

    # Define the Genetic Algorithm parameters
    POPULATION_SIZE = population_size
    MAX_GENERATIONS = max_generations
    MUTATION_RATE = mutation_rate
    stopping_criterion = POPULATION_SIZE // 3
    # list of all the source nodes
    #start_nodes = [node for node in nodes if node.source_location == SOURCE]

    start_nodes = [node for node in G.nodes if G.nodes[node]["source_location"] == SOURCE]

    population = initialize_population(POPULATION_SIZE, start_nodes)
    # Genetic Algorithm loop
    for generation in range(MAX_GENERATIONS):

        #parent1, parent2 = select_2_parents(population)
        parent1, parent2 = selection(population)
        #print(f"parent 1:  {[G.nodes[node]['ID'] for node in parent1]} path:{[G.nodes[node]['segment'] for node in parent1]} fitness={evaluate_chromosome((parent1))}")
        #print(f"parent 2:  {[G.nodes[node]['ID'] for node in parent2]} path:{[G.nodes[node]['segment'] for node in parent2]} fitness={evaluate_chromosome((parent2))}")
        child1 = crossover(parent1)
        child2 = crossover(parent2)
        #child1 , child2 = new_crossover(parent1,parent2)

        #print(f"child 1 : {[G.nodes[node]['ID'] for node in child1]}")
        #print(f"child 2 : {[G.nodes[node]['ID'] for node in child2]}")


        if child1 not in population.values() and len(child1)>0:
            population[len(population)] = child2
        if child2 not in population.values() and len(child2)>0:
            population[len(population)] = child2



    min = evaluate_chromosome(population[0])
    id = 0
    for i in range(1, len(population)):
        total_cost = evaluate_chromosome(population[i])
        if total_cost < min:
            id = i
            min = total_cost
    best_chromosome = population[id].copy()


    #print(f"population: {index},{[node.ID for node in best_chromosome]}")
    print(f"population: {id},{[node for node in best_chromosome]},total price = {min}")



def genetic1():
    SOURCE = "S"
    DESTINATION = "D"
    bidding = pd.read_csv("largeDS.csv")
    dataProcessing(bidding, SOURCE, DESTINATION)
    nodes, edges, G ,N= create_bidding_graph(bidding)
    #genetic_algo(10, 10, 0.1)

    start_time = time.time()
    genetic_algo(10, 10, 0.1)
    end_time = time.time()
    runtime = end_time - start_time
    print(f"runtime = {runtime:.4f} s")
    #
    n_iterations = 10
    attr = {genetic_algo:(10, 10, 0.1)}

    for algo in [genetic_algo]:
        runtimes = []
        mem_usages = []
        for i in range(n_iterations):
            print("#################################### {} ###################################".format(i))
            start_time = time.time()
            mem_usage = max(memory_usage((algo,attr[algo]), interval=0.1))
            end_time = time.time()
            runtime = end_time - start_time
            runtimes.append(runtime)
            mem_usages.append(mem_usage)
        avg_runtime = sum(runtimes) / len(runtimes)
        avg_mem_usage = sum(mem_usages) / len(mem_usages)
        print(f"Algorithm {algo.__name__}: average runtime = {avg_runtime:.4f} s, average memory usage = {avg_mem_usage:.4f} MB")






