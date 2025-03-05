from typing import List, Tuple, Optional, Union

import numpy as np
from qiskit import transpile
import matplotlib.pyplot as plt

SEED = 999

def mermin_max_value(num_qubits, backend):
    avg_fidelity = 1
    concurrence = 1
    ideal_value = 2**(num_qubits - 1)
    physical_qubits = BFS_GHZ(num_qubits, backend)[1]
    
    for q in physical_qubits:
        avg_fidelity *= (1 - backend.target['measure'][(q,)].error) * concurrence
    return ideal_value*avg_fidelity

def measure_time(function, *args, **kwargs):
    start_time = time.time()
    result = function(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time

def estimator_value(terms: List[str], coeffs: List[float]) -> float:
    """
    Computes the expectation value of a Mermin polynomial using the Estimator tool

    Args:
        terms (list): List of Mermin inequality terms (strings).
        coeffs (list): List of corresponding coefficients (floats).

    Returns:
        float: The computed expected value.
    """
    num_qubits = len(terms[0])
    qc = GHZ_state(num_qubits)
    obsv = SparsePauliOp(terms, coeffs)
    
    estimator = Estimator()
    job = estimator.run(qc, observables=obsv)
    exps = job.result().values[0]
    
    return exps


def get_metrics(circuits, backend=None, physical_qubits=[]):
    fidelity = 1
    if isinstance(circuits, list):
        circuits = circuits
    else:
        circuits = [circuits]

    num_circuits = len(circuits)
    
    if backend:
        circuits = transpile(circuits, backend, optimization_level=0, initial_layout=physical_qubits, seed_transpiler=SEED)
        for op in range(len(backend.operations)):
            if backend.operations[op].name in ['cz', 'cx', 'cy', 'ecr']:
                control_op = backend.operations[op].name

        # Some backends report the CNOT error for both directions of the edge, which can lead to inconsistencies.
        # If the number of entries exceeds a certain threshold we use a set with sorted tuples to ensure uniqueness.
        if len(backend.target[control_op].keys()) > backend.num_qubits*2:
            unique_edges = {tuple(sorted(edge)) for edge in backend.target[control_op]}
        else:
            unique_edges = backend.target[control_op].keys()
        
        fidelity *= np.prod([1 - backend.target[control_op][edge].error for edge in unique_edges if set(edge).issubset(physical_qubits)])
        fidelity *= np.prod([1 - backend.target['measure'][(q,)].error for q in physical_qubits])

    avg_depth = sum(circuit.depth() for circuit in circuits) / num_circuits
    avg_two_qubit_gates_depth = sum(circuit.depth(lambda x: len(x.qubits) == 2 and x.name != 'barrier') for circuit in circuits) / num_circuits

    return{
        'average_depth': avg_depth,
        'average_two_qubit_gates': avg_two_qubit_gates_depth,
        'fidelity': fidelity
    }