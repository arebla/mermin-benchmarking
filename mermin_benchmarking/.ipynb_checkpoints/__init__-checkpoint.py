# mermin_benchmarking/__init__.py

from .circuit_generation import (
    mermin_terms,
    simplified_mermin_terms,
    mermin_circuits,
    mermin_aspect_circuit,
    modified_mermin_aspect_circuit,
    bitstring2circuit,
)

from .ghz_optimization import (
    GHZ_state,
    W_state,
    get_physical_qubits,
    get_aux_physical_qubits,
    find_best_subgraph,
)

from .error_mitigation import(
    correlated_readout_error_matrix,
    corrected_counts,
)

from .measurement import(
    run_static,
    run_dynamic,
    MerminExperiment,
)

from .benchmarking import(
    estimator_value,
    get_metrics,
)

from .visualizations import(
    draw_circuits,
    matrix_representation,
)

from .deprecated import(
    required_memory,
    find_connected_subgraphs,
)

#__all__ = [
#    'mermin_terms',
#    'simplified_mermin_terms',
#]
