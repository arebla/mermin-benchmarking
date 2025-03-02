from typing import List, Tuple, Optional, Union
from itertools import combinations
import sys
import psutil
import math
import random

import numpy as np
import rustworkx as rx
from rustworkx import PyGraph
from rustworkx.visit import BFSVisitor
from qiskit import QuantumCircuit, transpile
from qiskit.providers.backend import Backend
from .ghz_optimization import TreeEdgesRecorder

def required_memory(qpu_size, subgraph_size):
    """
    Calculate the total memory required to store all combinations of subgraphs.

    Args:
        qpu_size (int): Number of qubits in the QPU.
        subgraph_size (int): Number of nodes/qubits in each subgraph.

    Returns:
        float: Total memory required in gigabytes (GB).
    """
    float_size = sys.getsizeof(0.0) # Size of a float in bytes
    int_size = sys.getsizeof(0) # Size of an integer in bytes
    tuple_overhead = sys.getsizeof(()) # Approximate overhead in bytes
    list_overhead = sys.getsizeof([]) # Approximate overhead in bytes

    num_combinations = math.comb(qpu_size, subgraph_size)
    one_combination_size = (int_size * subgraph_size) + float_size + tuple_overhead
    total_mem = (num_combinations * one_combination_size + list_overhead*2) / 1024**3

    return total_mem


def find_connected_subgraphs(G: PyGraph, size: int, iterations: int = 10**3) -> List[Tuple[int]]:
    """
    rustworkx recently added a connected_subgraphs function.
    ---
    Finds connected subgraphs of a given size in the graph.
    Uses different strategies based on available memory.

    Args:
        G (PyGraph): The graph to search for subgraphs.
        size (int): The size of the subgraphs to find.
        iterations (int, optional): Maximum number of iterations for the random strategy. Default is 10^3.
        
    Returns:
        List[Tuple[int]]: A list of nodes in each connected subgraph.
    """
    required_mem_gb = required_memory(len(G.nodes()), size)
    available_mem_gb = psutil.virtual_memory().available / (1024 ** 3) * 0.7

    if required_mem_gb < available_mem_gb:
        subgraphs = []
        for nodes in combinations(G.nodes(), size):
            subgraph = G.subgraph(nodes)
            if rx.is_connected(subgraph):
                subgraphs.append(nodes)
        
    else:
        subgraphs = set()

        for _ in range(iterations):
            start_node = random.choice(G.nodes())
        
            vis = TreeEdgesRecorder(size)
            rx.bfs_search(G, [start_node], vis)
        
            nodes = tuple(sorted(vis.nodes))
            subgraphs.add(nodes)
        subgraphs = list(subgraphs)

    return subgraphs