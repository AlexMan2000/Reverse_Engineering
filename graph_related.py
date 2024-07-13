import unittest
import networkx as nx
import queue
import matplotlib.pyplot as plt
from enum import Enum
from typing import Dict, List
from collections import defaultdict
import sys
import copy
import numpy as np


class ProblemType(Enum):
    TopologicalOrdering = 1
    EventualSafeStates = 2
    KeysAndRoom = 3
    FindIsland = 4
    FindBridges = 5
    FindArticulationPoint = 6

class Graph:
    """
    Simple graph data structure
    Doesn't allow parallel edges, but allow self-loops
    """

    def __init__(self, graph=None, format=None, type="uni"):
        self.numNode = 0
        self.edges = []
        self.data = defaultdict(list)
        self.type = type
        if graph is not None:
            assert format is not None
            if format == "edge":
                assert type is not None
            self.transform_graph(graph, format, type)

        self.start = 0

    def __len__(self):
        return self.numNode

    def __iter__(self):
        self.current = self.start
        # Preventing unexpected behaviors
        self.key_list = list(self.data.keys()).copy()
        return self

    def __getitem__(self, item):
        return self.data[item]

    def __next__(self):
        if self.current < self.numNode:
            current = self.key_list[self.current]
            self.current += 1
            return current
        else:
            raise StopIteration

    def transform_graph(self, graph, format="adj_map", type="di"):
        if format == "adj_list":
            for index, li in enumerate(graph):
                for v in li:
                    self.add_edge(index, v)
        elif format == "adj_map":
            for key, nei_list in graph.items():
                for v in nei_list:
                    self.add_edge(key, v)
        elif format == "edge":
            if type == "uni":
                for edge in graph:
                    self.add_edge(edge[0], edge[1])
                    self.add_edge(edge[1], edge[0])
            else:
                for edge in graph:
                    self.add_edge(edge[0], edge[1])

    def add_edge(self, u, v):
        if u not in self.data:
            self.numNode += 1
        if v not in self.data:
            self.numNode += 1
        self.data[u].append(v)
        self.data[v].extend([])
        if [u, v] in self.edges:
            raise RuntimeError("You have added this edge before")
        self.edges.append([u, v])

    def remove_edge(self, u, v):
        if u not in self.data or v not in self.data:
            raise RuntimeError("Edge doesn't exist!")
        self.data[u].remove(v)
        self.edges.remove([u, v])
        if self.type == "uni":
            self.data[v].remove(u)
            self.edges.remove([v, u])

    def remove_node(self, u):
        """
        Remove the node and all the edges associated to it from the graph
        """
        for v in self.data[u]:
            self.remove_edge(u, v)
        del self.data[u]
        self.numNode -= 1

    def remove_nodes(self, u_list):
        """
        Remove the node and all the edges associated to it from the graph
        """
        for u in u_list:
            self.remove_node(u)

    def flip_edge(self, u, v):
        self.remove_edge(u, v)
        self.add_edge(v, u)

    def reverse_graph(self):
        edges_copy = self.edges.copy()
        for edge in edges_copy:
            self.flip_edge(edge[0], edge[1])

    def find_all_SCCs(self):
        pass

    def find_all_CCs(self):
        pass

    def decompose_graph(self, list_nodes):
        pass

    def get_reversed_graph(self, graph):
        copied_graph = copy.deepcopy(graph)
        edges_copy = copied_graph.get_edges().copy()
        for edge in edges_copy:
            copied_graph.flip_edge(edge[0], edge[1])

        return copied_graph

    def find_all_neighbors(self, u):
        return self.data[u]

    def get_degree_map(self):
        node_degree_map = {}

        for node in self.data:
            node_degree_map[node] = {"in": 0, "out": 0}

        for node in self.data:
            for neighbor in self.data[node]:
                node_degree_map[node]["out"] += 1
                node_degree_map[neighbor]["in"] += 1

        return node_degree_map

    def get_adj_list(self):
        res_list = [[] for i in range(self.numNode)]
        for key, value in self.data.items():
            if not isinstance(key, int):
                raise RuntimeError("Non convertable!")
            res_list[key].extend(value)

        return res_list

    def get_adj_map(self):
        return self.data

    def get_adj_matrix(self):
        pass

    def get_edges(self):
        return self.edges

    def visualize_graph(self):
        if self.type == "di":
            G = nx.DiGraph()
        else:
            G = nx.Graph()

        for node in self.data:
            G.add_node(node)
            for neighbor in self.data[node]:
                G.add_edge(node, neighbor)

        pos = nx.planar_layout(G)

        nx.draw(G, pos, with_labels=True, node_size=700, node_color="lightblue", font_size=10, font_weight='bold',
                arrowsize=20, arrows=True)
        plt.show()


class GraphProblem:
    def __init__(self, graph: Graph, t: ProblemType):
        self.graph = graph
        self.type = t

    def getGraph(self) -> Graph:
        return self.graph

    def getType(self) -> ProblemType:
        return self.type


class GraphProblemSolver:
    def __init__(self, problem: GraphProblem):
        self.problem = problem
        self.graph = problem.getGraph().get_adj_map()

    def getFormattedGraph(self):
        return self.graph

    def topologicalSort(self, graph: Graph, mode="dfs", detectCycle=True):
        if mode == "dfs":
            return self.topologicalSortDFS(graph, detectCycle)
        else:
            return self.topologicalSortQueue(graph, detectCycle)

    def runProblem(self):
        self.graph = self.problem.getGraph()

        if self.problem.getType() == ProblemType.TopologicalOrdering:
            return self.topologicalSort(self.graph, mode="queue")
        elif self.problem.getType() == ProblemType.EventualSafeStates:
            reversed_graph = self.graph.get_reversed_graph(self.graph)
            res = self.topologicalSort(reversed_graph, mode="queue", detectCycle=False)
            return list(sorted(res))
        elif self.problem.getType() == ProblemType.KeysAndRoom:
            # res = self.topologicalSort(mode="queue", detectCycle=False)
            pass
            # return len(res) == len(self.graph)

    def topologicalSortDFS(self, graph, detectCycle=True):

        def dfsHelper(graph, visited, recStack, topoList, currNode):
            visited[currNode] = True

            # Used for cycle detection
            recStack[currNode] = True

            for successor in graph[currNode]:
                if not visited[successor]:
                    if dfsHelper(graph, visited, recStack, topoList, successor):
                        return True
                elif recStack[successor]:
                    return True

            recStack[currNode] = False
            topoList.append(currNode)

        visited = {}
        recStack = {}
        topoList = []

        for node in graph:
            visited[node] = False
            recStack[node] = False

        for node in graph:
            if not visited[node]:
                if dfsHelper(graph, visited, recStack, topoList, node):
                    if detectCycle:
                        raise LookupError("Cycle Detected")
                    else:
                        return ["Error"]

        return list(reversed(topoList))

    # Kahn's algorithm
    def topologicalSortQueue(self, graph: Graph, detectCycle=True):
        q = queue.Queue()
        topoList = []

        # 1. Find all the nodes that have no incoming edges(source nodes)
        nodeDegreeMap = graph.get_degree_map()

        sourceNodes = list(map(lambda elem: elem[0], filter(lambda elem: elem[1]["in"] == 0, nodeDegreeMap.items())))

        # print(sourceNodes)
        # 2. Put all the souce nodes into the queue
        for node in sourceNodes:
            q.put(node)

        # 3. While the queue is not empty
        while not q.empty():
            # 3.1 Pop the node from the front of the queue
            currNode = q.get()

            # 3.2 Append that source node to the topo ordering list
            topoList.append(currNode)

            # 3.3 For each neighbor of the current node
            for neighbor in graph[currNode]:
                # 3.3.1 Decrease the in degree of this neighbor by one(deleting the incoming edges)
                nodeDegreeMap[neighbor]["in"] -= 1

                # 3.3.2 If the neighbor becomes a source due to 0 in-degree, append it to the queue
                if nodeDegreeMap[neighbor]["in"] == 0:
                    q.put(neighbor)

        # 4. Finally, If the length of the topo ordering list is less than the number of nodes,
        # then there is a cycle in the graph and no topological ordering is possible
        # Since if there is a cycle, then the in-degree of the nodes in the cycle will always be
        # bigger than or equal to 1 and there will be no source nodes added to the queue and the
        # while loop is terminated prematurely.
        if len(topoList) < len(graph):
            if detectCycle:
                raise LookupError("Cycle Detected!")

        return topoList



    def find_critical_connections(self, algo_mode="tarjan") -> List[List[int]]:
        if algo_mode == "tarjan":
            res = self.tarJanAlgo(self.graph, type="Bridge")
        else:
            res = []
        return res

    def findAllSCCs(self, algo_mode="tarjan"):
        if algo_mode == "tarjan":
            return self.tarJanAlgo(self.graph, type="SCC")
        else:
            return self.kosaraJuAlgo(self.graph)

    # Tarjan's Algorithm
    def tarJanAlgo(self, graph, type):

        def dfsBridge(u, parent, disc, low):

            nonlocal time
            nonlocal bridges
            # 1. Once the node u is first visited, set the disc[u] and low[u] to 1st visited time
            disc[u] = time
            low[u] = time
            time += 1

            # 2. Start visit its neighbors
            for v in graph[u]:
                # Tree edge case: update low[u] = min(low[u], low[v]) after v has finished its DFS
                if disc[v] == -1:
                    dfsBridge(v, u, disc, low)
                    low[u] = min(low[u], low[v])

                    if low[v] > disc[u]:
                        bridges.append([u, v])
                elif v == parent:
                    continue
                else:
                    # Back edge case(visited and ancestor node): update low[u] = min(low[u], disc[v])
                    low[u] = min(low[u], disc[v])

            # if low[u] == disc[u] and disc[u] != 0:
            #     bridges.append([parent, u])

        def dfsSCCs(u, disc, low, ast, rst):

            nonlocal time
            nonlocal SCCs
            # 1. Once the node u is first visited, set the disc[u] and low[u] to 1st visited time
            disc[u] = time
            low[u] = time
            time += 1
            ast[u] = True
            rst.append(u)
            # 2. Start visit its neighbors
            for v in graph[u]:
                # Tree edge case: update low[u] = min(low[u], low[v]) after v has finished its DFS
                if disc[v] == -1:
                    dfsSCCs(v, disc, low, ast, rst)
                    low[u] = min(low[u], low[v])
                elif ast[v]:
                    # Back edge case(visited and ancestor node): update low[u] = min(low[u], disc[v])
                    low[u] = min(low[u], disc[v])

            # 3. u has done visited, and we need to see if this node is the start of an SCC
            if disc[u] == low[u]:
                temp = []
                # 3.1 We have to pop the node from the same group
                w = -1
                while w != u:
                    w = rst.pop()
                    temp.append(w)
                    ast[w] = False  # backtrack
                SCCs.append(temp)

        def dfsCut():
            pass


        numNode = len(graph)
        time = 0  # discover time ticker
        SCCs = []
        bridges = []

        # 1. Initialize
        # 1.1 Discover time and low-link values
        disc = [-1 for _ in range(numNode)]
        low = [-1 for _ in range(numNode)]


        # 1.2 Stack to keep track of the visiting stage, for fast access
        auxStack = [False for _ in range(numNode)]

        # 1.3 Real stack data structure
        realStack = []

        # 2. Start the DFS process
        for i in range(numNode):
            if disc[i] == -1:
                if type == "SCC":
                    dfsSCCs(i, disc, low, auxStack, realStack)
                elif type == "Bridge":
                    dfsBridge(i, -1, disc, low)

        print(disc)
        print(low)

        if type == "SCC":
            return SCCs
        elif type == "Bridge":
            return bridges

    def kosaraJuAlgo(self, graph):
        pass


if __name__ == "__main__":
    testGraphMulti = {"A": ["B", "C"],
                      "B": ["C", "D", "E"],
                      "C": ["F"],
                      "D": [],
                      "E": ["F"],
                      "F": []}
    testGraphEmpty = {

    }
    testGraphCycle = {
        "A": ["B", "C"],
        "B": ["C", "D", "E"],
        "C": ["D"],
        "D": [],
        "E": ["A"]
    }
    testGraphState = {
        "0": ["1", "2"],
        "1": ["2", "3"],
        "2": ["5"],
        "3": ["0"],
        "4": ["5"],
        "5": [],
        "6": []
    }
    testGraphRoom1 = {
        "0": ["1"],
        "1": ["2"],
        "2": ["3"],
        "3": []
    }
    testGraphRoom2 = {
        "0": ["1", "3"],
        "1": ["3", "0", "1"],
        "2": ["2"],
        "3": ["0"]
    }
    testGraphRoom3 = {
        "0": ["1"],
        "1": ["1"],

    }

    testGraphSCC1 = [[1, 0], [0, 2], [2, 1], [0, 3], [3, 4]]
    testGraphSCC2 = [[0, 1], [1, 2], [2, 3]]
    testGraphSCC3 = [[0, 1], [1, 2], [2, 0], [1, 3], [1, 4], [1, 6], [3, 5], [4, 5]]
    testGraphCC1 = [[0,1],[1,2],[2,0],[1,3]]
    testGraphCC2 = [[0, 1], [1, 2], [2, 3]]


    g = Graph(testGraphCC1, format="edge", type="uni")
    g.visualize_graph()
    # print(g.data)
    p1 = GraphProblem(g, ProblemType.EventualSafeStates)
    p2 = GraphProblem(g, ProblemType.KeysAndRoom)

    gs = GraphProblemSolver(p2)
    print(gs.find_critical_connections())
