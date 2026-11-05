"""Small exact one-to-one assignment for <=6 slots, without scipy."""
from itertools import permutations
import numpy as np


def assignment(cost, miss_penalty=40.):
    """Return (true_index, slot_index, cost) matches and unmatched true ids.

    Cost is [true objects, predicted slots]; no arbitrary ordering assumption.
    An object can remain unmatched when all alternatives exceed the penalty.
    This is an exact small-scene diagnostic, not a differentiable loss.
    """
    matrix=np.asarray(cost,float)
    if (matrix.ndim!=2 or max(matrix.shape)>6 or matrix.shape[0]<1
            or np.any(~np.isfinite(matrix)) or miss_penalty<=0):
        raise ValueError("Expected finite small cost matrix")
    n,m=matrix.shape
    # Dummy columns represent independently unmatched ground-truth objects.
    augmented=np.concatenate((matrix,np.full((n,n),miss_penalty)),axis=1)
    candidates=permutations(range(m+n),n)
    chosen=min(candidates,key=lambda p:sum(augmented[i,j] for i,j in enumerate(p)))
    matches=[(i,j,float(matrix[i,j])) for i,j in enumerate(chosen) if j<m]
    missing=[i for i,j in enumerate(chosen) if j>=m]
    return matches,missing
