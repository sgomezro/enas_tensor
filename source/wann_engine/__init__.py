"""
Tensorized Weight Agnostic Neural Networks (WANN) engine.

Reference: A. Gaier and D. Ha, "Weight Agnostic Neural Networks", NeurIPS 2019.

Everything that runs per generation lives on one device (GPU if available):
  * population    -> padded tensors  adj (P,N,N) bool, act (P,N) long,
                     rank (P,N) float64, active (P,N) bool       (_population.py)
  * forward pass  -> one batched GEMM per depth level, shared weight factored out
                                                                  (_forward.py)
  * mutation      -> vectorized, sync-free (scatter/gather); DAG guaranteed by
                     only allowing edges from lower to higher node rank (_variation.py)
  * ranking       -> Pareto (non-dominated) sort on (mean, max) or
                     (mean, -connections), alternating as in WANN  (_selection.py)
  * training      -> per-edge weights learned by backprop          (_training.py)

Node layout per genome (N = max_nodes slots):
  [0]                bias (constant 1)
  [1 .. n_obs]       observations
  [n_in .. n_in+n_out-1]  outputs      (rank 1.0)
  [n_in+n_out .. N-1]     hidden slots (inactive until an add-node mutation)
  inputs have rank 0.0; hidden nodes get a rank strictly between their endpoints.
"""
from ._activations import ACT_NAMES, K_ACT, SHARED_WEIGHTS, apply_activations
from ._population import (Compiled, Population, compile_population, compute_depth,
                          describe, init_population)
from ._forward import reference_forward, wann_forward
from ._variation import mutate
from ._selection import next_generation, pareto_fronts
from ._training import (LOGIT_CLAMP, ce_acc, logits_edges, logits_shared,
                        test_acc_edges, train_edges)
