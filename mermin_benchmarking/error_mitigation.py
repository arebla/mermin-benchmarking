from typing import List, Tuple, Optional, Union

import numpy as np
from scipy.optimize import minimize
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.providers.backend import Backend
from qiskit_ibm_runtime import (
    SamplerV2 as Sampler,
    SamplerOptions
)

from .circuit_generation import bitstring2circuit


def correlated_readout_error_matrix(physical_qubits: List[int], backend: Backend):
    """
    Returns the readout mitigation matrix (inverse of the assignment matrix) given a list of physical qubits and a backend.

    Args:
        physical_qubits (List[int]): List of physical qubits for initial layout.
        backend (Backend): Backend object for execution.
        
    Returns:
        numpy.ndarray: The mitigation matrix.

    """
    if backend is None:
        return None
        
    sampler = Sampler(mode=backend)
    
    shots = 1024*2**len(physical_qubits)
    aux_circuits = []

    num_qubits = len(physical_qubits)
    bitstrings = [bin(i)[2:].zfill(num_qubits) for i in range(2**num_qubits)]

    for bitstring in bitstrings:
        aux_circuits.append(bitstring2circuit(bitstring))
        
    transpiled_circuits = transpile(aux_circuits, backend, optimization_level=0, initial_layout=physical_qubits)
    result = sampler.run(transpiled_circuits, shots=shots).result()
    counts = [result[i].data.meas.get_counts() for i in range(len(aux_circuits))]
        
    df = pd.DataFrame(counts, columns=bitstrings).fillna(0) / shots
    assignment_matrix = np.transpose(df.to_numpy())   
    mitigation_matrix = np.linalg.inv(assignment_matrix)

    return mitigation_matrix 


def corrected_counts(raw_data, readout_matrix):
    """
    Corrects raw counts using the given readout matrix.

    Args: 
        raw_data (numpy.ndarray): The raw count data.
        readout_matrix (numpy.ndarray): The mitigation matrix.
    
    Returns:
        numpy.ndarray: The corrected count data.
    """
    num_qubits = int(np.log2(len(readout_matrix)))
    raw_data = raw_data.copy()
    assignment_matrix = np.linalg.inv(readout_matrix)
    
    def RRS(x, data_idx):
        residuals = raw_data[data_idx] - np.dot(assignment_matrix, x)
        return np.sum(residuals**2)
        
    for data_idx in range(len(raw_data)):
        shots = np.sum(raw_data[data_idx])
        x0 = np.random.rand(2**num_qubits)
        x0 = x0 / np.sum(x0)
        cons = {'type': 'eq', 'fun': lambda x: shots - np.sum(x)}
        bnds = [(0,shots) for _ in x0]
        res = minimize(RRS, x0, args=(data_idx,), method='SLSQP', constraints=cons, bounds=bnds, tol=1e-24)
        raw_data[data_idx] = res.x
        
    return raw_data