import numpy as np
from spp_lcp import spp_lcp  # assumes you saved the function in spp_lcp.py

def main():
    # Example: 2-variable LCP
    #   find z >= 0, w = q + M z >= 0, zᵀ w = 0
    #
    # Choose an M and q such that solution is easy to verify.
    # Here: M = [[2, -1], [-1, 2]], q = [-1, -1]
    M = np.array([[2.0, -1.0],
                  [-1.0, 2.0]])
    q = np.array([-1.0, -1.0])

    # Cold-start (diodes OFF by default if q >= 0, otherwise heuristic)
    on, info = spp_lcp(M, q, on_init=None, tau=1e-10)

    print("=== LCP test ===")
    print("M =\n", M)
    print("q =", q)
    print("Active set (ON):", on)
    print("Info:", info)

    # Verify: compute (z,w)
    B = on
    N = ~on
    z = np.zeros_like(q)
    if B.any():
        z[B] = np.linalg.solve(M[np.ix_(B,B)], -q[B])
    w = q + M @ z
    print("z =", z)
    print("w =", w)
    print("Complementarity z·w =", np.dot(z,w))

if __name__ == "__main__":
    main()
