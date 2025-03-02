from typing import List, Tuple, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit.providers.backend import Backend
from .ghz_optimization import GHZ_state, W_state

def mermin_terms(n: int) -> Tuple[List[str], List[float]]:
    """
    Generates the Mermin inequality terms and coefficients of size n.

    Args:
        n (int): The number of qubits.

    Returns:
        tuple: A tuple containing the list of terms (xys) and 
               corresponding coefficients.
    """
    bins = [bin(i)[2:].zfill(n) for i in range(2**n)]
    cs = [np.real((1j**b.count('1') - np.conj(1j**b.count('1'))) / 2j)
          for b in bins]
      
    xys = [''.join('Y' if b == '1' else 'X' for b in bins[i])
           for i in range(len(bins)) if cs[i]!=0]
      
    coefs = [cs[i]
           for i in range(len(cs)) if cs[i]!=0]

    return xys, coefs


def simplified_mermin_terms(n: int) -> Tuple[List[str], List[float]]:
    """
    Generates the simplified Mermin inequality terms and coefficients based on symmetries.

    Args:
        n (int): The number of qubits.

    Returns:
        tuple: A tuple containing the simplified list of terms (xys) 
               and corresponding coefficients.
    """
    string_list, coeff_list = mermin_terms(n)
    y_counts = {}
    
    for string, coeff in zip(string_list, coeff_list):
        y_count = string.count('Y')
        y_counts[y_count] = y_counts.get(y_count, 0) + coeff
    
    xys = [('Y'*y_count).rjust(n, 'X') 
           for y_count in y_counts.keys()]
    coeffs = list(y_counts.values())
    
    return xys, coeffs

def mermin_circuits(terms: List[str], backend: Optional[Backend] = None, 
                    physical_qubits: Optional[List[int]] = None) -> List[QuantumCircuit]:
    """
    Generates a Quantum circuit for each Mermin inequality term.

    Args:
        terms (list): List of Mermin inequality terms (strings).
        backend (Optional[Backend], optional): Backend for execution (default None).
        physical_qubits (Optional[List[int]], optional): List of physical qubits. 
                                                         Required if backend is provided.

    Returns:
        list: List of QuantumCircuit objects representing the Mermin circuits.
    """
    circuits = []
    num_qubits = len(terms[0])
    measure_list = [num for num in range(num_qubits)] 
    
    for current_term in terms:
        qc = GHZ_state(num_qubits, backend, physical_qubits)
        
        # Add relative phase |ψ> = 1/sqrt(2) * (|0...> + i |1...>)
        qc.s(0)
        qc.barrier()
        for char in range(num_qubits):
            if current_term[char] == 'X':
                qc.h(char)
            if current_term[char] == 'Y':
                qc.sdg(char)
                qc.h(char)
        qc.barrier() 
        qc.measure(measure_list, measure_list)
        circuits.append(qc)

    return circuits
    

def mermin_aspect_circuit(num_qubits: int, backend: Optional[Backend] = None, 
                          physical_qubits: Optional[List[int]] = None) -> QuantumCircuit:
    """
    Generates a quantum circuit for a dynamic Mermin-like experiment.

    Args:
        num_qubits (int): Number of qubits.
        backend (Optional[Backend], optional): Backend for execution (default None).
        physical_qubits (Optional[List[int]], optional): List of physical qubits. 
                                                         Required if backend is provided.

    Returns:
        QuantumCircuit: The QuantumCircuit object representing the dynamic Mermin circuit.
    """
    qc = QuantumCircuit(2*num_qubits, 2*num_qubits)

    if backend:
        physical_qubits = physical_qubits[:num_qubits]
    qc.compose(GHZ_state(num_qubits, backend, physical_qubits), [i for i in range(num_qubits)], inplace=True)
    
    # Add relative phase |ψ> = 1/sqrt(2) * (|0...> + i |1...>)
    qc.s(0)
    qc.barrier()
    for i in range(num_qubits, 2*num_qubits):
        qc.h(i)
        qc.measure(i, i)
        with qc.if_test((i, 0)): #X equiv 0
            qc.h(i - num_qubits)
        with qc.if_test((i, 1)): #Y equiv 1
            qc.sdg(i - num_qubits)
            qc.h(i - num_qubits)
        qc.measure(i - num_qubits, i - num_qubits)    

    return qc   


def modified_mermin_aspect_circuit(num_qubits: int, backend: Optional[Backend] = None, 
                                   physical_qubits: Optional[List[int]] = None) -> QuantumCircuit:
    """
    Generates a quantum circuit for a dynamic Mermin-like experiment using a W state in the ancilla qubits.

    Args:
        num_qubits (int): Number of qubits.
        backend (Optional[Backend], optional): Backend for execution (default None).
        physical_qubits (Optional[List[int]], optional): List of physical qubits. 
                                                         Required if backend is provided.

    Returns:
        QuantumCircuit: The QuantumCircuit object representing the dynamic Mermin circuit.
    """
    qc = QuantumCircuit(2*num_qubits, 2*num_qubits)

    if backend:
        physical_qubits = physical_qubits[:num_qubits]
    qc.compose(GHZ_state(num_qubits, backend, physical_qubits), [i for i in range(num_qubits)], inplace=True)

    # Add relative phase |ψ> = 1/sqrt(2) * (|0...> + i |1...>)
    qc.s(0)
    qc.compose(W_state(num_qubits), [i for i in range(num_qubits, 2*num_qubits)], inplace=True)

    for i in range(num_qubits, 2*num_qubits):
        qc.measure(i, i)
        with qc.if_test((i, 0)): #X equiv 0
            qc.h(i - num_qubits)
        with qc.if_test((i, 1)): #Y equiv 1
            qc.sdg(i - num_qubits)
            qc.h(i - num_qubits)
        qc.measure(i - num_qubits, i - num_qubits)    

    return qc

def bitstring2circuit(bitstring: str):
    """
    Generates a quantum circuit that initializes qubits based on a given bitstring.

    Args:
        bitstring (str): A string of '0' and '1' characters representing the desired qubit states.

    Returns:
        QuantumCircuit: The QuantumCircuit object representing the bitstring with measurements applied.
    """
    num_qubits = len(bitstring)
    qc = QuantumCircuit(num_qubits)
    [qc.x(num_qubits - 1 - bit) for bit in range(num_qubits) if bitstring[bit] == '1']
    
    qc.measure_all()
    return qc