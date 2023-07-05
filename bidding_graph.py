from collections import defaultdict
import networkx as nx

# GLOBAL VARIABLE
edges = []
nodes = []
G = nx.DiGraph()
# class BiddingNode:

class BiddingNode:
    def __init__(self, ID, pickup_time, delivery_time, price, source_location, destination_location, isSource, isTarget,
                 segment, distance_to_target):
        self.__dict__.update(locals())
        self.neighbors = []

    def addNeighbors(self, neighbor):
        self.neighbors.append(neighbor)

    def getNeighborsByID(self):
        return [neg.ID for neg in self.neighbors]


def create_nodes(bidding):
    nodes_by_source = defaultdict(list)  # Dictionary to store nodes by source_location
    for index, row in bidding.iterrows():
        attributes = {
            'ID': row["carrier id"],
            'pickup_time': row['pickup time'],
            'delivery_time': row['delivery time'],
            'price': row["price"],
            'source_location': row["origin"],
            'destination_location': row["dest"],
            'isSource': row["source?"],
            'isTarget': row["target?"],
            'segment': row['Segment'],
            'distance_to_target': row['distance_to_target']
        }
        node = BiddingNode(**attributes)
        nodes.append(node)
        nodes_by_source[node.source_location].append(node)
        node_attributes = {k: v for k, v in node.__dict__.items() if k != 'self'}
        G.add_node(node.ID, **node_attributes)

    node = BiddingNode('start', '00:00:00', '00:00:00', 0, 'start', 'start', False, False, 'start', 0)
    nodes.append(node)
    nodes_by_source[node.source_location].append(node)
    node_attributes = {k: v for k, v in node.__dict__.items() if k != 'self'}
    G.add_node(node.ID, **node_attributes)

    node = BiddingNode('target', '00:00:00', '00:00:00', 0, 'target', 'target', False, False, 'target', 0)
    nodes.append(node)
    nodes_by_source[node.source_location].append(node)
    node_attributes = {k: v for k, v in node.__dict__.items() if k != 'self'}
    G.add_node(node.ID, **node_attributes)

    return nodes_by_source

def create_edges(bidding):
    for index, row in bidding.iterrows():
        for index, row2 in bidding.iterrows():

            if row["dest"] == row2["origin"] and row["delivery time"] <= row2["pickup time"]:
                edges.append((row["carrier id"], row2["carrier id"]))
                G.add_edge(row["carrier id"], row2["carrier id"])
    for node in nodes:
        # add edges from the dummy node 'start' to all sources nodes
        if node.isSource:
            edges.append(('start', node.ID, node.price))
            G.add_edge('start', node.ID)
        # add edges from the dummy node 'target' to all target nodes
        if node.isTarget:
            edges.append([node.ID, 'target', 0])
            G.add_edge(node.ID, 'target')


def create_bidding_graph(bidding):

    nodes_by_source = create_nodes(bidding)
    create_edges(bidding)

    # add neighbors to all nodes
    n = {node.ID: node for node in nodes}
    for e in edges:
        if e[1] not in n[e[0]].neighbors:
            n[e[0]].addNeighbors(n[e[1]])

    return nodes, edges, G, nodes_by_source
