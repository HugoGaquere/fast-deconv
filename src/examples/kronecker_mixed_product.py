import numpy as np

def main():
    # Fix seed for reproducibility
    rng = np.random.default_rng(42)

    # Dimensions:
    # A: (m x n),  C: (n x r)
    # B: (p x q),  D: (q x s)
    m, n, r =  10,  10, 20
    p, q, s = 601, 601, 60

    # Random matrices
    A = rng.standard_normal((m, n))
    C = rng.standard_normal((n, r))
    B = rng.standard_normal((p, q))
    D = rng.standard_normal((q, s))

    # Explicit Kroneckers then matmul: (A⊗B)(C⊗D)
    kron_AB = np.kron(A, B)      # shape: (m*p, n*q)
    kron_CD = np.kron(C, D)      # shape: (n*q, r*s)
    left = kron_AB @ kron_CD     # shape: (m*p, r*s)

    # Mixed-product: (AC)⊗(BD)
    AC = A @ C                   # shape: (m, r)
    BD = B @ D                   # shape: (p, s)
    right = np.kron(AC, BD)      # shape: (m*p, r*s)

    breakpoint()

    # Check equality
    equal = np.allclose(left, right, rtol=1e-10, atol=1e-12)
    max_abs_diff = np.max(np.abs(left - right))

    print("Shapes:")
    print(f"(A ⊗ B)(C ⊗ D): {left.shape}")
    print(f"(AC) ⊗ (BD):   {right.shape}")
    print()
    print("allclose:", equal)
    print("max |difference|:", max_abs_diff)

    # Optional: assert if you want a hard check
    assert equal, "Mixed-product property failed (within numerical precision)."

if __name__ == "__main__":
    main()
