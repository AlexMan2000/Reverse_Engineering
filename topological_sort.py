
import unittest
import networkx as nx
import queue
import matplotlib.pyplot as plt
from enum import Enum
from typing import Dict, List


class ProblemType(Enum):
    TopologicalOrdering = 1
    EventualSafeStates = 2
    KeysAndRoom = 3
    FindIsland = 4

class GraphProblem:
    def __init__(self, graph: Dict, t: ProblemType):
        self.graph = graph
        self.type = t

    def getGraph(self):
        return self.graph

    def getType(self):
        return self.type

class GraphProblemSolver:

    def __init__(self, graph: Dict = {}):
        self.graph = graph


    def topologicalSort(self, mode="dfs", detectCycle = True):
        if mode == "dfs":
            return self.topologicalSortDFS(self.graph, detectCycle)
        else:
            return self.topologicalSortQueue(self.graph, detectCycle)


    def runProblem(self, problem: GraphProblem):
        self.graph = problem.getGraph()

        if problem.getType() == ProblemType.TopologicalOrdering:
            return self.topologicalSort(mode="queue")
        elif problem.getType() == ProblemType.EventualSafeStates:
            self.graph = self.flipGraph(self.graph)
            res = self.topologicalSort(mode="queue", detectCycle=False)
            return list(sorted(res))
        elif problem.getType() == ProblemType.KeysAndRoom:
            # res = self.topologicalSort(mode="queue", detectCycle=False)
            pass
            return len(res) == len(self.graph)


    def visualizeGraph(self, graph):
        G = nx.DiGraph()
        for node in self.graph:
            G.add_node(node)
            for neighbor in self.graph[node]:
                G.add_edge(node, neighbor)

        pos = nx.shell_layout(G)

        nx.draw(G, pos, with_labels=True, node_size = 700, node_color ="lightblue",font_size=10, font_weight='bold', arrowsize=20)
        plt.show()

    def topologicalSortDFS(self, graph, detectCycle=True):
        visited = {}
        recStack = {}
        topoList = []

        for node in graph:
            visited[node] = False
            recStack[node] = False

        for node in graph:
            if not visited[node]:
                if self.dfsHelper(graph, visited, recStack, topoList, node):
                    if detectCycle:
                        raise LookupError("Cycle Detected")
                    else:
                        return ["Error"]

        return list(reversed(topoList))


    def dfsHelper(self, graph, visited, recStack, topoList, currNode):
        visited[currNode] = True

        # Used for cycle detection
        recStack[currNode] = True

        for successor in graph[currNode]:
            if not visited[successor]:
                if self.dfsHelper(graph, visited, recStack, topoList, successor):
                    return True
            elif recStack[successor]:
                return True

        recStack[currNode] = False
        topoList.append(currNode)


    def topologicalSortQueue(self, graph, detectCycle = True):
        q = queue.Queue()
        topoList = []

        # 1. Find all the nodes that have no incoming edges(source nodes)
        nodeDegreeMap = self.findDegrees(graph)

        sourceNodes = list(map(lambda elem: elem[0], filter(lambda elem: elem[1]["in"] == 0,nodeDegreeMap.items() )))

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
            for neighbor in self.graph[currNode]:
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


    def findDegrees(self, graph) -> Dict:
        nodeDegreeMap = {}

        for node in graph:
            nodeDegreeMap[node] = {"in": 0, "out": 0}

        for node in graph:
            for neighbor in graph[node]:
                nodeDegreeMap[node]["out"] += 1
                nodeDegreeMap[neighbor]["in"] += 1

        return nodeDegreeMap

    def flipGraph(self, graph) -> Dict:
        res = {}
        for node in graph:
            res[node] = []

        for node in graph:
            for neighbor in graph[node]:
                res[neighbor].append(node)

        return res

    def dfsVisited(self, graph) -> Dict:
        pass

if __name__ == "__main__":
    testGraphMulti = {"A": ["B","C"],
                  "B": ["C","D","E"],
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

    g = GraphProblemSolver()
    p1 = GraphProblem(testGraphState, ProblemType.EventualSafeStates)
    p2 = GraphProblem(testGraphRoom3, ProblemType.KeysAndRoom)
    print(g.runProblem(p2))



