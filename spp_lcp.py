import numpy as np
from scipy.sparse import isspmatrix, csr_matrix
from scipy.sparse.linalg import spsolve

def spp_lcp(M, q, on_init=None, tau=1e-10, max_pivots=None, pivot_rule="first"):
    """
    Simple principal-pivoting to determine the complementarity pattern (active set).

    Parameters
    ----------
    M : (n,n) array_like or scipy.sparse matrix
        LCP matrix.
    q : (n,) array_like
        LCP right-hand side.
    on_init : (n,) bool array or None
        Warm start: True for indices initially in B (z>0,w=0 = diode ON).
        If None, starts from B = { i | q[i] < 0 }.
    tau : float
        Nonnegativity tolerance. Only pivot if z_i < -tau or w_i < -tau.
    max_pivots : int or None
        Hard cap on pivot count. Default = 10*n.
    pivot_rule : {"first","bland"}
        Deterministic tie-breaker; both behave the same here (lowest index).

    Returns
    -------
    on : (n,) bool array
        True where diode is ON (z>0, w=0). False where diode is OFF (z=0, w>0).
    info : dict
        Diagnostics with keys:
        - "pivots": number of pivots performed
        - "feasible": bool
        - "last_violation": ("z", idx) or ("w", idx) or None
    """
    # Normalize inputs
    q = np.asarray(q, dtype=float).ravel()
    if isspmatrix(M):
        M = M.tocsr()
    else:
        M = csr_matrix(np.asarray(M, dtype=float))
    n = q.size
    if M.shape != (n, n):
        raise ValueError("M must be square with size matching q")

    if on_init is None:
        on = q < 0.0  # cheap cold-start heuristic
    else:
        on = np.asarray(on_init, dtype=bool).copy()
        if on.size != n:
            raise ValueError("on_init length mismatch")

    if max_pivots is None:
        max_pivots = 10 * n

    pivots = 0
    last_violation = None

    # Pre-allocate work vectors
    z = np.zeros(n, dtype=float)
    w = np.zeros(n, dtype=float)

    def pick_first(mask):
        idx = np.nonzero(mask)[0]
        return None if idx.size == 0 else int(idx[0])

    while True:
        B = on
        N = ~on

        # Solve for z_B from q_B + (M_BB) z_B = 0; set z_N = 0
        if B.any():
            M_BB = M[B][:, B]
            rhs = -q[B].copy()
            try:
                z_B = spsolve(M_BB, rhs)
            except Exception as e:
                raise RuntimeError(f"Linear solve failed: {e}")
            z[:] = 0.0
            z[B] = z_B
        else:
            z[:] = 0.0

        # Compute w
        w[:] = q
        if B.any():
            w += M[:, B] @ z[B]

        # Check for violations
        viol_z = np.zeros(n, dtype=bool)
        if B.any():
            viol_z[B] = z[B] < -tau
        viol_w = np.zeros(n, dtype=bool)
        if N.any():
            viol_w[N] = w[N] < -tau

        if not (viol_z.any() or viol_w.any()):
            last_violation = None
            break

        i_z = pick_first(viol_z)
        i_w = pick_first(viol_w)

        if i_z is not None:
            on[i_z] = False
            last_violation = ("z", i_z)
        elif i_w is not None:
            on[i_w] = True
            last_violation = ("w", i_w)
        else:
            break

        pivots += 1
        if pivots >= max_pivots:
            break

    info = {"pivots": pivots, "feasible": last_violation is None, "last_violation": last_violation}
    return on, info
