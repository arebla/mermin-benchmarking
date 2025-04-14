from typing import List, Tuple, Optional, Union

import numpy as np
import random
import rustworkx as rx
from rustworkx import PyGraph
from rustworkx.visit import BFSVisitor
from qiskit import QuantumCircuit, transpile
from qiskit.providers.backend import Backend

SEED = 999

class TreeEdgesRecorder(BFSVisitor):
    def __init__(self, max_size=None):
        self.edges = []
        self.nodes = set()
        self.max_size = max_size

    def tree_edge(self, edge):
        self.edges.append(edge)

    def discover_vertex(self, v):
        if self.max_size is None or len(self.nodes) < self.max_size:
            self.nodes.add(v)
        return self.max_size is None or len(self.nodes) < self.max_size

def find_best_subgraph(subgraphs: List[Tuple[int]], backend: Backend) -> Tuple[int]:
    """
    Finds the nodes of the subgraph with the highest fidelity score, considering both measurement errors 
    and control gate errors, from a list of subgraphs.

    Args:
        subgraphs (List[Tuple[int]]): Subgraphs to evaluate. 
        backend (Backend): Backend to use for fidelity scoring.

    Returns:
        Tuple[int]: Nodes of the subgraphs with the highest fidelity scores. 
    """
    fidelity_scores = []

    # Search through backend operations to find the control operation
    for op in range(len(backend.operations)):
        if backend.operations[op].name in ['cz', 'cx', 'cy', 'ecr']:
            control_op = backend.operations[op].name

    # Extract unique edges for the control operation error
    # Some backends report the CNOT error for both directions of the edge, which can lead to inconsistencies.
    # If the number of entries exceeds a certain threshold we use a set with sorted tuples to ensure uniqueness.
    if len(backend.target[control_op].keys()) > backend.num_qubits*2:
        unique_edges = {tuple(sorted(edge)) for edge in backend.target[control_op]}
    else:
        unique_edges = backend.target[control_op].keys()

    
    for subgraph_nodes in subgraphs:
        # Measure error contribution to score
        score = np.prod([1 - backend.target['measure'][(node,)].error for node in subgraph_nodes])

        # Control gates error contribution to score
        filtered_edges = [t for t in unique_edges if t[0] in subgraph_nodes and t[1] in subgraph_nodes]
        for edge in filtered_edges:
            score *= 1 - backend.target[control_op][edge].error
        
        fidelity_scores.append(score)
    
    max_score_index = np.argmax(fidelity_scores)
    best_subgraph_nodes = subgraphs[max_score_index]

    return best_subgraph_nodes


def get_physical_qubits(num_qubits: int, backend: Optional[Backend] = None) -> List[int]:
    """
    Retrieves a physical layout of qubits on the given backend that optimizes connectivity.

    Args:
        num_qubits (int): Number of qubits required for the circuit.
        backend (Optional[Backend], optional): Backend object for execution. Default is None.

    Returns:
        List[int]: List corresponding to the nodes of the subgraph
            with the highest fidelity score.
    """
    if backend is None:
        return []

    num_qubits_backend = backend.num_qubits
    coupling_map = backend.coupling_map
    coupling_map_edges = list(coupling_map.get_edges())

    G = rx.PyGraph()
    G.add_nodes_from(range(num_qubits_backend))
    
    for edge in coupling_map_edges:
        G.add_edge(edge[0], edge[1], None)

    if num_qubits > 17: # Change if more than ~20 GB of RAM
        subgraphs = set()
        iterations = 10**3
        
        for _ in range(iterations):
            start_node = random.choice(G.nodes())
        
            vis = TreeEdgesRecorder(num_qubits)
            rx.bfs_search(G, [start_node], vis)
        
            nodes = tuple(sorted(vis.nodes))
            subgraphs.add(nodes)
        subgraphs = list(subgraphs)

    else:
        subgraphs = rx.connected_subgraphs(G, num_qubits)
        
    best_subgraph = find_best_subgraph(subgraphs, backend)
    
    physical_qubits = sorted(list(best_subgraph))

    return physical_qubits


def get_aux_physical_qubits(num_qubits: int, backend: Optional[Backend] = None, physical_qubits: List[int] = []) -> List[int]:
    """
    Retrieves a physical layout of qubits on the given backend that optimizes connectivity.

    Args:
        num_qubits (int): Number of qubits required for the circuit.
        physical_qubits (List[int]): Primary physical qubits.
        backend (Optional[Backend], optional): Backend object representing the quantum device. Default is None.

    Returns:
        List[int]: List corresponding to the nodes of
            the subgraph with the highest fidelity score
            that does not have nodes in common with the primary physical qubits.
    """
    if backend is None:
        return []

    num_qubits_backend = backend.num_qubits
    coupling_map = backend.coupling_map
    coupling_map_edges = list(coupling_map.get_edges())

    G = rx.PyGraph()
    G.add_nodes_from(range(num_qubits_backend))
    
    for edge in coupling_map_edges:
        G.add_edge(edge[0], edge[1], None)

    G.remove_nodes_from(physical_qubits)
    subgraphs = rx.connected_subgraphs(G, num_qubits)
    best_subgraph = find_best_subgraph(subgraphs, backend)
    
    aux_physical_qubits = sorted(list(best_subgraph))

    return aux_physical_qubits


def BFS_GHZ(num_qubits: int, backend: Backend, physical_qubits: List[int]) -> QuantumCircuit:
    """
    Generates a GHZ state using BFS algorithm on the coupling map.

    Args:
        num_qubits (int): Number of qubits.
        backend (Backend): The backend to use for execution.
        physical_qubits (List[int], optional): List of physical qubits.

    Returns:
        QuantumCircuit: The quantum circuit with minimum depth.
    """
    transpiled_depths = []
    circuits = []

    num_qubits_backend = backend.num_qubits
    coupling_map = backend.coupling_map
    coupling_map_edges = list(coupling_map.get_edges())

    G = rx.PyGraph()
    G.add_nodes_from(range(num_qubits_backend))
    
    for edge in coupling_map_edges:
        G.add_edge(edge[0], edge[1], None)

    # When creating the subgraph physical_qubits must be sorted
    best_subgraph = G.subgraph(physical_qubits)
    
    for i in range(num_qubits):
        root_vertex = [i]
        vis = TreeEdgesRecorder()
        rx.bfs_search(best_subgraph, root_vertex, vis)

        qc = QuantumCircuit(num_qubits, num_qubits)

        qc.h(root_vertex)
        for edge in vis.edges:
            u, v, _ = edge
            qc.cx(u, v)

        qc_transpiled = transpile(qc, backend, optimization_level=0, initial_layout=physical_qubits, seed_transpiler=SEED)
        circuits.append(qc)
        transpiled_depths.append(qc_transpiled.depth())

    min_depth = np.argmin(transpiled_depths)

    return circuits[min_depth]
    

def GHZ_state(num_qubits: int, backend: Optional[Backend] = None, physical_qubits: List[int] = []) -> QuantumCircuit:
    """
    Generates a Quantum Circuit object representing an n-qubit GHZ state.

    Args:
        num_qubits (int): Number of qubits.
        backend (Optional[Backend], optional): Backend object for execution. Default is None.
        physical_qubits (List[int]): List of physical qubits. Required if backend is provided.

    Returns:
        QuantumCircuit: Quantum circuit of a GHZ state.
    """
    qc = QuantumCircuit(num_qubits, num_qubits)
    k = 0
    temp_n = num_qubits
    
    qc.h(0)

    if backend and not physical_qubits:
        raise ValueError("Physical qubits must be provided if a backend is specified.")
        
    if backend:
        # Use BFS algorithm if backend is provided
        qc = BFS_GHZ(num_qubits, backend, physical_qubits)
    else: 
        while temp_n > 0:
            temp_n //= 2
            k += 1
            
        for j in range(k, 0, -1):
            round_half_up = np.floor(((num_qubits-1)/2**j) + 0.5).astype(int)
            for i in range(round_half_up):
               qc.cx(2**j*i, 2**j*i + 2**(j-1))        

    return qc


def W_state(num_qubits: int) -> QuantumCircuit:
    """
    Returns a QuantumCircuit object representing the n-qubit W state.

    Args:
        num_qubits (int): Number of qubits in the W state.

    Returns:
        QuantumCircuit: Quantum circuit representing the W state.
    """
    qc = QuantumCircuit(num_qubits, num_qubits)

    qc.x(0)

    #if num_qubits % 2 == 1:
    #    # Odd: W_state - |11..1>
    #    for i in range(1, num_qubits):
    #        θ = 2 * np.arccos(np.sqrt(1 / (num_qubits - i + 2)))
    #        qc.cu(θ, 0, 0, 0, i - 1, i)
    #        qc.cx(i, i - 1)

    #    θ = 2 * np.arccos(np.sqrt(1 / 2))
    #    qc.cu(θ, np.pi, 0, 0, num_qubits - 1, num_qubits - 2)

    #    for i in range(0, num_qubits - 2):
    #        qc.ccx(num_qubits - 1, num_qubits - 2, i)

    #else:
        # Even: W_state
    for i in range(1, num_qubits):
        θ = 2 * np.arccos(np.sqrt(1 / (num_qubits - i + 1)))
        qc.cu(θ, 0, 0, 0, i - 1, i)
        qc.cx(i, i - 1)

    return qc