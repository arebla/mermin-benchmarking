from typing import List, Tuple, Optional, Union
import os
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import RuntimeEncoder, RuntimeDecoder, QiskitRuntimeService

from .circuit_generation import mermin_terms, simplified_mermin_terms
from .benchmarking import get_metrics
from .measurement import counts2staticvalue, counts2dynamicvalue

def matrix_representation(matrix: np.ndarray):
    """
    Visualizes a given matrix as a heatmap.
    
    Args:
        matrix (np.ndarray): A square matrix.

    Returns:
        None: Displays a heatmap.
    """
    n = int(np.log2(len(matrix)))
    #matrix = correlated_readout_error_matrix(physical_qubits, backend)
    
    labels = [bin(i)[2:].zfill(n) for i in range(2**n)]
    labels = [fr'$|{label}\rangle$' for label in labels]
    diff = np.abs(matrix - np.eye(2**n))

    heatmap = plt.matshow(np.log(diff + 1), cmap=plt.cm.Reds)
    #heatmap = plt.matshow(np.log(diff + 1e-10), cmap=plt.cm.Reds)
    
    #heatmap = plt.matshow(matrix, vmin=-1, vmax=1, cmap="bwr")
    #heatmap = plt.matshow(diff, cmap=plt.cm.Reds, clim=[0, 0.2])
    plt.colorbar(heatmap, fraction=0.0467, pad=0.02)
    plt.xticks(np.arange(len(labels)), labels, rotation='vertical')
    plt.yticks(np.arange(len(labels)), labels)
    plt.title('Prepared state', fontsize=14)
    plt.ylabel('Measured state', fontsize=14)
    plt.show()

def draw_circuits(circuits: Union[QuantumCircuit, List[QuantumCircuit]], terms: Optional[List[str]] = None, scale: Optional[float] = None):
    """
    Draws one or more quantum circuits with optional labels.

    Args:
        circuits (Union[QuantumCircuit, List[QuantumCircuit]]): 
            A single quantum circuit or a list of quantum circuits.
        terms (Optional[List[str]], optional): 
            A list of string labels corresponding to the circuits.
        scale (Optional[float], optional): 
            Scaling factor for the circuit visualization.

    Returns:
        None: Displays the circuit plots.
    """
    if isinstance(circuits, QuantumCircuit):
        return circuits.draw(scale=scale)
    
    if len(circuits) == 1:
        return circuits[0].draw()

    if terms is None:
        terms = ['' for _ in range(len(circuits))]
        
    num_cols = int(np.log2(len(circuits)))
    num_rows = int(np.log2(len(circuits)))

    while num_cols*num_rows < len(circuits):
        num_cols += 1
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(6*num_rows, 3*num_rows))

    axs = axs.flatten()
    
    for i, circuit in enumerate(circuits):
        circuit_fig = circuit.draw(output='mpl', style='iqp')
    
        canvas = FigureCanvas(circuit_fig)
        canvas.draw()
        image = canvas.renderer.buffer_rgba()
        
        axs[i].imshow(image)
        axs[i].set_title(f'{terms[i]}')
        axs[i].axis('off')

    # Hide extra plots
    for j in range(len(circuits), len(axs)):
        axs[j].axis('off')
        
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0, hspace=0)
    plt.show()

def load_files(directory: str):
    """
    Given a directory with files named in the format "backend-experiment_type-num_qubits-timestamp", returns a nested dictionary of Mermin values organized by backend, experiment type, and number of qubits.

    Args:
        directory (str): Path of the folder where the .json files are stored.
    Returns:
        collections.defaultdict: Dictionary with the Mermin values by backend, experiment and number of qubits.
    
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for file_name in os.listdir(directory):
        if file_name.endswith(('.json')):  
            parts = file_name.split('-')
            backend_name = parts[0]
            experiment_type = parts[1]
            num_qubits = parts[2]
            timestamp = parts[3]  

            file_path = os.path.join(directory, file_name)

            with open(file_path, "r") as file:
                result = json.load(file, cls=RuntimeDecoder)

                if num_qubits == '24q' or (num_qubits == '9q' and experiment_type == 'dynamic'):
                    continue # It takes too long ^_^
                    
                if experiment_type == 'static':
                    value = counts2staticvalue(result)
                    data[backend_name][experiment_type][num_qubits].append(value)
                    
                elif experiment_type == 'static_sym':
                    value = counts2staticvalue(result, symmetries=True)
                    data[backend_name][experiment_type][num_qubits].append(value)
                    
                elif experiment_type == 'dynamic':
                    if len(result) == 1:
                        value = counts2dynamicvalue(result)
                        data[backend_name][experiment_type][num_qubits].append(value)
                    else:
                        for res in result:
                            value = counts2dynamicvalue([res])
                            data[backend_name][experiment_type][num_qubits].append(value)

    return data

def generate_job_files(job_id: str, service: QiskitRuntimeService):
    """
    Given a IBM Quantum job id yields two files: a .json with counts data and a .txt with information about the job.

    Args:
        job_id (str): The IBM Quantum job ID.
    Returns: 
        None: Saves the corresponding files in the ./data/ folder. 
    """
    
    job = service.job(job_id)
    job_result = job.result()

    num_qubits = job.result()[0].data.c.num_bits
    
    # Get type of experiment
    if len(job.result()) == len(mermin_terms(num_qubits)[0]):
        job_mode = 'static'
    elif len(job.result()) == len(simplified_mermin_terms(num_qubits)[0]):
        job_mode = 'static_sym'
    else:
        job_mode = 'dynamic'
    
    file_name = (
        f"./data/{job.backend().name}-"
        f"{job_mode}-{num_qubits}q-{job.creation_date.strftime('%Y%m%d%H%M%S')}.json"
    )
    with open(file_name, "w") as file:
        json.dump(job.result(), file, cls=RuntimeEncoder)
    
    # Save job details to disk
    job_details = []
    
    job_details.append(f"Creation Date: {job.creation_date.strftime('%Y-%m-%d %H:%M:%S')}")
    job_details.append(f"Job ID: {job.job_id()}")
    job_details.append(f"Backend: {job.backend().name}")
    job_details.append(f"Options: {job.inputs['options']}")
    job_details.append(f"Usage Estimation: {job.usage_estimation}")
    job_details.append(f"Usage: {job.usage()}")
    job_details.append(f"Qiskit Version: {job.metrics()['qiskit_version']}")
    
    
    job_details.append(f"Number of qubits: {job.result()[0].data.c.num_bits}")
    job_details.append(f"Number of circuits: {len(job.result())}")
    job_details.append(f"Shots per circuit: {job.result()[0].data.c.num_shots}")
    
    
    job_circuits = [job.inputs['pubs'][c][0] for c in range(len(job.inputs['pubs']))]
    job_circuits_metrics = get_metrics(job_circuits)
    job_details.append(f"Avg circuit depth: {job_circuits_metrics['average_depth']}")
    job_details.append(f"Avg number of 2 qubit gates: {job_circuits_metrics['average_two_qubit_gates']}")
    
    job_details.append(f"Pubs Layout: {job.inputs['pubs'][0][0].layout.initial_layout}")
    
    
    file_name_details = (
        f"./data/{job.backend().name}-"
        f"{job_mode}-{num_qubits}q-{job.creation_date.strftime('%Y%m%d%H%M%S')}.txt"
    )
    with open(file_name_details, 'w') as f:
        for line in job_details:
            f.write(line + '\n')