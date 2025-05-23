from typing import List, Tuple, Optional, Union
import os
import json
from collections import defaultdict
import csv
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import RuntimeEncoder, RuntimeDecoder, QiskitRuntimeService, IBMBackend
from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2

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


def load_mermin_values(directory: str, index_path: Optional[str] = None):
    """
    Given a directory with files named in the format "backend-experiment_type-num_qubits-timestamp.json", 
    returns a nested dictionary of Mermin values organized by backend, experiment type, and number of qubits.

    Args:
        directory (str): Path of the folder where the .json files are stored.
        index_path (str): Path to the index file.
    Returns:
        collections.defaultdict: Dictionary with the Mermin values by backend, experiment and number of qubits.
    """
    
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    if index_path:
        with open(index_path, "r") as file:
            index = json.load(file)
    else:
        index = None
    
    for file_name in os.listdir(directory):
        if file_name.endswith(('.json')):
            parts = file_name.split('-')
            if len(parts) < 4:
                continue
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
                    if index:
                        mitigation_matrix = fetch_mitigation_matrix(index, file_name)
                        value = counts2staticvalue(result, readout_matrix=mitigation_matrix)
                    else:
                        value = counts2staticvalue(result)
                    data[backend_name][experiment_type][num_qubits].append(value)
                    
                elif experiment_type == 'static_sym':
                    if num_qubits in ['4q', '7q'] and index:
                        mitigation_matrix = fetch_mitigation_matrix(index, file_name)
                        value = counts2staticvalue(result, symmetries=True, readout_matrix=mitigation_matrix)
                    else:
                        value = counts2staticvalue(result, symmetries=True)
                    data[backend_name][experiment_type][num_qubits].append(value)
                    
                elif experiment_type == 'dynamic':
                    if len(result) == 1:
                        if num_qubits in ['3q'] and index:
                            mitigation_matrix = fetch_mitigation_matrix(index, file_name)
                            value = counts2dynamicvalue(result, readout_matrix=mitigation_matrix)
                        else:
                            value = counts2dynamicvalue(result)
                        data[backend_name][experiment_type][num_qubits].append(value)
                    else:
                        for res in result:
                            if num_qubits in ['3q'] and index:
                                mitigation_matrix = fetch_mitigation_matrix(index, file_name)
                                value = counts2dynamicvalue([res], readout_matrix=mitigation_matrix)
                            else:
                                value = counts2dynamicvalue([res])
                            data[backend_name][experiment_type][num_qubits].append(value)

    def defaultdict2dict(d):
        if isinstance(d, defaultdict):
            d = {j: defaultdict2dict(k) for j, k in d.items()}
        return d
    data = defaultdict2dict(data)
    return data


def fetch_mitigation_matrix(index, file_name):
    """
    Given a dictionary of data and a specific file name,
    extracts the mitigation matrix from the corresponding calibration file.

    Args:
        index (dict): The dictionary containing the data.
        file_name (str): The name of the file (e.g., 'ibm_sherbrooke-static-4q-20250405030228.json').

    Returns:
        list: The mitigation matrix if found, or None if not found.
    """
    parts = file_name.split('-')
    backend_name = parts[0]
    if backend_name in index:
        for calibration_file, nested_dict in index[backend_name].items():
            for subkey, calibration_data in nested_dict.items():
                if file_name in calibration_data['files']:
                    return np.array(calibration_data['mitigation_matrix'])

    return None
    

def generate_job_files(job_id: str, service: QiskitRuntimeService):
    """
    Generate two files for a given IBM Quantum job: a JSON file with result counts 
    and a text file with job metadata.

    Args:
        job_id (str): The IBM Quantum job ID.
        service (QiskitRuntimeService): The Qiskit service instance used to retrieve the job.
        
    Returns: 
        None: Saves the resulting files in the "./data/" folder. 
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
        num_qubits /= 2
    
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
    print(f"Data successfully saved to {job_mode}-{num_qubits}q-{job.creation_date.strftime('%Y%m%d%H%M%S')}")


# Save calibration data from IBM Quantum devices
# Heavily inspired by https://quantumcomputing.stackexchange.com/questions/40011/how-to-download-historical-calibration-data-from-ibm-quantum-devices
def save_calibration_to_csv(backend: IBMBackend, date: datetime):
    """
    Save the calibration data of an IBM backend at a specified date to a CSV file.
    
    Args:
        backend (IBMBackend): The IBM Quantum backend to extract calibrations from.
        date (datetime): A datetime object representing the desired calibration date.

    Returns:
        None: Writes a .CSV file to the "./data/backend_calibrations/" directory.

    Usage:
    ```
    from qiskit_ibm_runtime import QiskitRuntimeService
    from datetime import datetime

    service = QiskitRuntimeService(
        channel='ibm_quantum',
        instance='ibm-q/open/main',
        token='# Insert token here',
    )

    backend = service.backend("ibm_sherbrooke")
    date = datetime(day=14, month=3, year=2025, hour=16)

    save_calibration_to_csv(backend, date)
    ```

    Notes: 
        - This function is hardcoded to work with 'Eagle r3' processor types.
        - The CSV structure is designed to be compatible with the `refresh_backend_from_csv`
          function and follows the same format as the calibrations provided by the 
          IBM Quantum platform.
    """
    
    properties = backend.properties(datetime=date)
    
    date_str = date.strftime('%Y%m%d%H%M%S')
    filename = f"./data/backend_calibrations/{backend.name}-calibrations-{date_str}.csv"
    
    fieldnames = [
        "Qubit", "T1 (us)", "T2 (us)", "Frequency (GHz)", "Anharmonicity (GHz)",
        "Readout assignment error ", "Prob meas0 prep1 ", "Prob meas1 prep0 ",
        "Readout length (ns)", "ID error ", "Z-axis rotation (rz) error ",
        "√x (sx) error ", "Pauli-X error ", "ECR error ", "Gate time (ns)", "Operational"
    ]
    
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
    
        for idx, qubit in enumerate(properties.qubits):
            id_error = rz_error = sx_error = x_error = ecr_error = gate_time = ""
    
            for op in backend.operation_names:
                if op == "id":
                    id_error = backend.target['id'][(idx,)].error
                elif op == "rz":
                    rz_error = backend.target['rz'][(idx,)].error
                elif op == "sx":
                    sx_error = backend.target['sx'][(idx,)].error
                elif op == "x":
                    x_error = backend.target['x'][(idx,)].error
                elif op == "ecr":
                    ecr_error_list = []
                    gate_time_list = []
                    
                    for (q1, q2), instr_props in backend.target["ecr"].items():
                        if q1 == idx:
                            ecr_error_list.append(f"{q1}_{q2}:{instr_props.error}")
                            gate_time_list.append(f"{q1}_{q2}:{int(instr_props.duration * 1e9)}")  # ns
            

            def get_qubit_prop(qubit, name):
                for prop in qubit:
                    if prop.name == name:
                        return prop.value
                return ""
            
            writer.writerow({
                "Qubit": idx,
                "T1 (us)": get_qubit_prop(qubit, 'T1'),
                "T2 (us)": get_qubit_prop(qubit, 'T2'),
                "Frequency (GHz)": get_qubit_prop(qubit, 'frequency'),
                "Anharmonicity (GHz)": get_qubit_prop(qubit, 'anharmonicity'),
                "Readout assignment error ": get_qubit_prop(qubit, 'readout_error'),
                "Prob meas0 prep1 ": get_qubit_prop(qubit, 'prob_meas0_prep1'),
                "Prob meas1 prep0 ": get_qubit_prop(qubit, 'prob_meas1_prep0'),
                "Readout length (ns)": get_qubit_prop(qubit, 'readout_length'),
                "ID error ": id_error,
                "Z-axis rotation (rz) error ": rz_error,
                "√x (sx) error ": sx_error,
                "Pauli-X error ": x_error,
                "ECR error ": ";".join(ecr_error_list),
                "Gate time (ns)": ";".join(gate_time_list),
                "Operational": "true"
            })

    print(f"Data successfully saved to {filename}")


# Adapted from: https://egrettathula.wordpress.com/2023/01/10/refreshable-fake-backends/
def refresh_backend_from_csv(self, calibrations_csv: str):
    """
    Load calibration data from a CSV file into the FakeBackendV2 backend.

    Args:
        calibrations_csv (str): Path to the CSV file containing calibration data.

    Returns:
        None

    Usage:
    ```
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane
    backend = FakeBrisbane()

    backend.refresh_from_csv(calibrations_csv)
    ```
    
    Notes: 
        - These `prop_names`, `gate_names` and row format may need to be adjusted 
          to load calibrations from processor types other than 'Eagle r3'.
        - This function does not update the calibration date.
    """
    
    if self._props_dict is None:
        self._set_props_dict_from_json()
 
    prop_names = ['T1', 'T2', 'frequency', 'anharmonicity', 'readout_error', 
                  'prob_meas0_prep1', 'prob_meas1_prep0', 'readout_length']
    gate_names = ['id', 'rz', 'sx', 'x']

    # Change calibration date if needed
    #self._props_dict['last_update_date'] = datetime.utcnow().isoformat() + 'Z'
    
    with open(calibrations_csv) as csvfile:
        reader = csv.reader(csvfile)
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            index = int(row[0])

            # Set the `prop_names` errors
            for node in self._props_dict['qubits'][index]:
                for prop_index, prop_name in enumerate(prop_names):
                    if node['name'] == prop_name:
                        value = row[prop_index + 1]
                        if value.strip() != '':
                            node['value'] = float(value)
                        else:
                            node['value'] = 0.0
 
            # Single-qubit gate errors
            for node in self._props_dict['gates']:
                if node['qubits'] == [index]:
                    for gate_index, gate_name in enumerate(gate_names):
                        if node['gate'] == gate_name:
                            for param in node['parameters']:
                                if param['name'] == 'gate_error':
                                    param['value'] = float(row[gate_index + 9])
 
            # CNOT error
            for s in row[13].split(';'):
                if ':' not in s:
                    continue
                key, val = s.strip().split(':')
                idxs = key.split('_')
                for node in self._props_dict['gates']:
                    if node['qubits'] == [int(idxs[0]), int(idxs[1])] and node['gate'] == 'ecr':
                        for param in node['parameters']:
                                if param['name'] == 'gate_error':
                                    param['value'] = float(val)
 
            # CNOT gate time
            for s in row[14].split(';'):
                if ':' not in s:
                    continue
                key, val = s.strip().split(':')
                idxs = key.split('_')
                for node in self._props_dict['gates']:
                    if node['qubits'] == [int(idxs[0]), int(idxs[1])] and node['gate'] == 'ecr':
                        for param in node['parameters']:
                                if param['name'] == 'gate_length':
                                    param['value'] = float(val)
 
setattr(FakeBackendV2, 'refresh_from_csv', refresh_backend_from_csv) 
