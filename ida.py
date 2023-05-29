import pandas as pd
import datetime
import time



def dataProcessing(bidding, source, target):
    # convert the start and end times to time objects
    start = []
    end = []
    source_nodes = []
    target_nodes = []
    for index, row in bidding.iterrows():
        # extract the start and end times from the string
        time_str_list = row["pickup time"].strip("[]").split("-")
        end_str_list = row["delivery time"].strip("[]").split("-")

        start_time_str = time_str_list[0]
        end_time_str = end_str_list[0]

        # use strptime() to convert the start and end times to time objects
        start_time_obj = datetime.datetime.strptime(start_time_str, "%H:%M:%S").time()
        end_time_obj = datetime.datetime.strptime(end_time_str, "%H:%M:%S").time()
        start.append(start_time_obj)
        end.append(end_time_obj)

    # add the start and end time objects to the data frmae
    bidding["pickup time"] = start
    bidding["delivery time"] = end

    # Marking bidding starting from the source point and/or target point
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


edges = []
nodes=[]


def create_edges(origin_group, dest_group):
    # create a list to store the pairs of indices that satisfy the condition

    for i in range(len(origin_group)):
        for j in range(len(dest_group)):
            # check if delivery time in first row index is equal or less than pickup time in second row index
            if dest_group.iloc[j]['delivery time'] <= origin_group.iloc[i]['pickup time']:
                edges.append((dest_group.iloc[j]['carrier id'], origin_group.iloc[i]['carrier id']))


def group_by(bidding):
    grouped_by_dest = bidding.groupby("dest")
    grouped_by_origin = bidding.groupby("origin")

    for group_name, group_df in grouped_by_origin:

        for group_name2, group_dest in grouped_by_dest:
            # group_name is a tuple of the group names (i.e. the unique values in the columns)
            # group_df is a DataFrame containing all the rows with the same values in the columns
            if group_name == group_name2:
                create_edges(group_df, group_dest)


class BiddingNode:
    def __init__(self, ID, pickup_time, delivery_time,price, source_location, destination_location,isSource,isTarget):
        self.ID = ID
        self.pickup_time = pickup_time
        self.delivery_time = delivery_time
        self.price = price
        self.source_location = source_location
        self.destination_location = destination_location
        self.isSource = isSource
        self.isTarget = isTarget
        self.neighbors = []

    def addNeighbors(self,neighbor):
        self.neighbors.append(neighbor)

def create_nodes(bidding):
    # Print the nodes and their attributes

    for index, row in bidding.iterrows():

        nodes.append(BiddingNode(row["carrier id"],row['pickup time'],row['delivery time'], row["price"], row["origin"], row["dest"],row["source?"],row["target?"]) )
    return nodes


import sys

#
# def dijkstra_node_weights( source):
#     #  Initialize distances and visited set
#     n = {node.ID: node for node in nodes}
#     distances = {node.ID: sys.maxsize for node in nodes}
#
#     distances[source] = 0
#     visited = set()
#     previous = {node.ID: None for node in nodes}
#
#
#     while len(visited) != len(nodes):
#         #  Find the node with the smallest distance
#         unvisited = {node: distances[node] for node in distances if node not in visited}
#         current_node = min(unvisited, key=unvisited.get)
#
#         #  Mark the current node as visited
#         visited.add(current_node)
#
#         #  Update distances for each neighbor of the current node
#         for neighbor in n[current_node].neighbors:
#             tentative_distance = distances[current_node] + n[neighbor].price
#             if tentative_distance < distances[neighbor]:
#                 distances[neighbor] = tentative_distance
#                 previous[neighbor] = current_node
#
#     return distances, previous

############################# IDA STAR ####################################

def is_dest(node):
    return nodes[node].isTarget


def h(node):
    return nodes[node].price


def IDA_star( max_price, s, d):
    threshold = h( s)
    path = ['start']

    t = 0
    while t is not None:
        t = search( d, path, h( s), threshold, max_price)
        if t == "FOUND":
            return path, threshold
        if t > max_price:
            t = None
        threshold = t
        if threshold > max_price:
            t = None
    return t


def search( d, path, g, threshold, max_price):


    node = path[-1]
    f = g
    if f > threshold:
        return f
    if is_dest( node):
        return "FOUND"
    min_price = max_price + 1
    for succ in nodes[node].neighbors:

        if succ not in path:
            path.append(succ)
            f_succ = g + h( succ)
            t = search( d, path, f_succ, threshold, max_price)

            if t == "FOUND":
                return "FOUND"
            if t < min_price:
                min_price = t
            path.pop()
    return min_price
from pytictoc import TicToc
if __name__ == "__main__":
    bidding = pd.read_csv("bidding.csv")
    print(bidding.tail(15))
    dataProcessing(bidding, "S", "D")
    group_by(bidding)
    create_nodes(bidding)


    for node in nodes:
        # add edges from the dummy node 'start' to all sources nodes
        if node.isSource:
            edges.append(('start', node.ID))
        # add edges from the dummy node 'target' to all target nodes
        elif node.isTarget:

            edges.append((node.ID, 'target'))

    nodes.append(BiddingNode('start','00:00:00','00:00:00' ,0,'start','start', False, False))
    nodes.append(BiddingNode('target', '00:00:00', '00:00:00', 0,'target','target', False, False))

    n = {node.ID: node for node in nodes}

    #add neighbors to all nodes
    for e in edges:
        if e[1] not in n[e[0]].neighbors:
            n[e[0]].addNeighbors(n[e[1]].ID)

    st_dij = time.process_time_ns()

    #distances, previous = dijkstra_node_weights( 'start')
    # target_node = 'target'
    # path = [target_node]
    # while previous[target_node] is not None:
    #     path.insert(0, previous[target_node])
    #     target_node = previous[target_node]
    # print(distances['target'])
    # print(path)

    # ed_dij = time.process_time_ns()
    # t_dij = ed_dij - st_dij
    nodes = n

    st_ida= time.process_time_ns()


    # sum_x = 0
    # for i in range(1000000):
    #     sum_x += i
    print(IDA_star( 10000, 'start', 'target'))
    #time.sleep(10)
    ed_ida=time.process_time_ns()

    t_ida=ed_ida-st_ida

    t = TicToc()  # create TicToc instance
    t.tic()  # Start timer

    print(t.toc() ) # Print elapsed time
    print(IDA_star(10000, 'start', 'target'))
    print("CPU execution time: ", end="\n")
    print("IDA ", t_ida)
   # print("dij ", t_dij)
