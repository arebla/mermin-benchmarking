from typing import List, Tuple, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from qiskit import QuantumCircuit

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