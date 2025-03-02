from typing import List, Tuple, Optional, Union, Dict

SEED = 999

from qiskit import QuantumCircuit, transpile
from qiskit.providers.backend import Backend
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import (
    QiskitRuntimeService, 
    SamplerV2 as Sampler,
    SamplerOptions
)

import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

from .circuit_generation import mermin_terms, simplified_mermin_terms, mermin_circuits, modified_mermin_aspect_circuit, mermin_aspect_circuit
from .ghz_optimization import get_physical_qubits, get_aux_physical_qubits
from .error_mitigation import correlated_readout_error_matrix, corrected_counts
from .benchmarking import get_metrics


class MerminExperiment:
    def __init__(self, num_qubits: int, shots: int = 1024, backend: Optional[Backend] = None, mode: str = 'static', symmetries: bool = False,
                error_mitigation: bool = True, sampler_options: Optional[Union[Dict, SamplerOptions]] = None):
        self.num_qubits = num_qubits
        self.shots = shots
        self.backend = backend
        self.mode = mode
        self.symmetries = symmetries
        self.terms, self.coeffs = self.terms_coeffs()
        self.physical_qubits = get_physical_qubits(self.num_qubits, self.backend)
        self.circuits = self.get_circuits()
        self.error_mitigation = error_mitigation
        self.readout_error_matrix = self.apply_error_mitigation()
        self.sampler_options = sampler_options
        
    def terms_coeffs(self) -> Tuple[List[str], List[float]]:
        if self.symmetries:
            terms, coeffs = simplified_mermin_terms(self.num_qubits)
        else:
            terms, coeffs = mermin_terms(self.num_qubits)
            
        return terms, coeffs

    def get_circuits(self) -> Union[List[QuantumCircuit], QuantumCircuit]:
        if self.mode == 'static':
            self.circuits = mermin_circuits(self.terms, self.backend, self.physical_qubits)
        elif self.mode == 'dynamic':
            self.physical_qubits = self.physical_qubits + get_aux_physical_qubits(self.num_qubits, self.backend, self.physical_qubits)
            self.circuits = mermin_aspect_circuit(self.num_qubits, self.backend, self.physical_qubits)
        else:
            raise ValueError("Mode must be either 'static' or 'dynamic'")
        return self.circuits
            
    def run(self):
        if self.mode == 'static':
            return run_static(self.circuits, self.shots,
                              self.backend, readout_matrix=self.readout_error_matrix, 
                              physical_qubits=self.physical_qubits, sampler_options=self.sampler_options)
        elif self.mode == 'dynamic':
            return run_dynamic(self.circuits, self.shots, 
                               self.backend, readout_matrix=self.readout_error_matrix, 
                               physical_qubits=self.physical_qubits, sampler_options=self.sampler_options)
        else:     
            raise ValueError("Mode must be either 'static' or 'dynamic'")
            
    def apply_error_mitigation(self):
        if self.backend == None:
            self.readout_error_matrix = None
        elif self.error_mitigation == True:
            self.readout_error_matrix = correlated_readout_error_matrix(self.physical_qubits, self.backend)
        else:
            self.readout_error_matrix = None
        return self.readout_error_matrix
        
    @property
    def metrics(self):
        return get_metrics(self.circuits, self.backend, self.physical_qubits)



def run_static(circuits: List[QuantumCircuit], shots: int = 1024, backend: Optional[Backend] = None,
               readout_matrix: Optional[np.ndarray] = None, physical_qubits: Optional[List[int]] = None, 
               sampler_options: Optional[Union[Dict, SamplerOptions]] = None) -> float:
    """
    Executes static quantum circuits and computes an observable value.

    Args:
        num_qubits (int): Number of qubits.
        circuits (List[QuantumCircuit]): Quantum circuits to execute.
        shots (int): Number of shots (measurement repetitions) to perform.
        coeffs (List[int]): Coefficients corresponding to each Mermin term for computing M_values.
        backend (Optional[Backend], optional): Backend object for execution. Default is None (uses Aer simulator).
        readout_matrix (Optional[np.ndarray], optional): Readout error mitigation matrix. Default is None.
        physical_qubits (Optional[List[int]], optional): List of physical qubits for initial layout. Required if backend is provided.
        sampler_options (Optional[Union[Dict, SamplerOptions]], optional): Sampler primitive options to use. Default is None.
    
    Returns:
        float: Computed observable value.

    """
    num_qubits = circuits[0].num_qubits
    coeffs = mermin_terms(num_qubits)[1]
    
    M_values = 0
    bitstrings = [bin(i)[2:].zfill(num_qubits) for i in range(2**num_qubits)]
    
    if backend:
        sampler = Sampler(mode=backend, options=sampler_options)
        circuits = transpile(circuits, backend, optimization_level=0, initial_layout=physical_qubits, seed_transpiler=SEED)
    else:
        aer_sim = AerSimulator()
        sampler = Sampler(mode=aer_sim)
        
    #result = backend.run(circuits, shots=shots).result()
    #counts = result.get_counts()
    result = sampler.run(circuits, shots=shots).result()
    counts = [result[i].data.c.get_counts() for i in range(len(circuits))]
    
    df = pd.DataFrame(counts, columns=bitstrings).fillna(0)

    # Apply Readout Error Mitigation (QREM)
    if readout_matrix is not None:
        results_matrix = df.to_numpy()
        results_matrix = corrected_counts(results_matrix, readout_matrix)
#        for i in range(len(circuits)):
        #    results_matrix[i] = np.transpose(np.dot(readout_matrix, np.transpose(results_matrix[i])))
        df = pd.DataFrame(results_matrix, columns=df.columns)   
    
    # Apply correction based on parity of '1's in bitstring columns
    for i in range(len(df.columns)):
        if df.columns[i].count('1') % 2 == 1:
            df[df.columns[i]] = df[df.columns[i]].apply(lambda x: x*(-1))

    for row in range(len(circuits)):
        M_values += sum(df.iloc[row]*coeffs[row]) / shots

    return M_values


def run_dynamic(circuit: QuantumCircuit, shots: int, backend: Optional[Backend] = None,
               readout_matrix: Optional[np.ndarray] = None, physical_qubits: Optional[List[int]] = None,
               sampler_options: Optional[Union[Dict, SamplerOptions]] = None) -> float:
    """
    Executes a dynamic quantum circuit and computes an observable value.

    Args:
        num_qubits (int): Number of qubits in the circuit.
        circuit (QuantumCircuit): Quantum circuit to execute.
        shots (int): Number of shots (measurement repetitions) to perform.
        backend (Optional[Backend], optional): Backend object for execution. Default is None (uses Aer simulator).
        readout_matrix (Optional[np.ndarray], optional): Readout error mitigation matrix. Default is None.
        physical_qubits (Optional[List[int]], optional): List of physical qubits for initial layout. Required if backend is provided.
        sampler_options (Optional[Union[Dict, SamplerOptions]], optional): Sampler primitive options to use. Default is None.

    Returns:
        float: Computed observable value.

    """
    M_values = 0
    num_qubits = int(circuit.num_qubits/2)
    terms, coeffs = mermin_terms(num_qubits)
    bitstrings = [bin(i)[2:].zfill(num_qubits*2) for i in range(2**(num_qubits*2))]
    
    if backend:
        sampler = Sampler(mode=backend, options=sampler_options)
        circuit = transpile(circuit, backend, optimization_level=0, initial_layout=physical_qubits, seed_transpiler=SEED)
    else:
        aer_sim = AerSimulator()
        sampler = Sampler(mode=aer_sim)
            
    result = sampler.run([circuit], shots=shots).result()
    counts = result[0].data.c.get_counts()

    df = pd.DataFrame(counts, index=[0], columns=bitstrings).fillna(0)

    # Apply Readout Error Mitigation (QREM)
    if readout_matrix is not None:
        results_matrix = df.to_numpy()
        results_matrix = corrected_counts(results_matrix, readout_matrix)
#        results_matrix = np.transpose(np.dot(readout_matrix, np.transpose(results_matrix)))
        df = pd.DataFrame(results_matrix, columns=df.columns)
    
    binary_terms = [term.replace('X', '0').replace('Y', '1') for term in terms]
    
    for i in range(len(terms)):
        columns_to_modify = df.columns[df.columns.str.contains(fr'^{binary_terms[i]}\d+$')]
        df[columns_to_modify] = df[columns_to_modify].apply(lambda x: x*(-1) if x.name[len(binary_terms[i]):].count('1') % 2 == 1 else x)
        df[columns_to_modify] = df[columns_to_modify].apply(lambda x: x*coeffs[i])
    
    M_values = df.iloc[0].sum() / (shots / 2**(num_qubits - 1))
    
    return M_values
