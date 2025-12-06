import cupy as cp

def main():
    rng = cp.random.default_rng(42)

    # Dimensions:
    # A: (m x n),  C: (n x r)
    # B: (p x q),  D: (q x s)
    m, n, r =  10,  10, 20
    p, q, s = 601, 601, 60

    # Random complex64 matrices
    A = rng.standard_normal((m, n), dtype=cp.float32) + 1j * rng.standard_normal((m, n), dtype=cp.float32)
    C = rng.standard_normal((n, r), dtype=cp.float32) + 1j * rng.standard_normal((n, r), dtype=cp.float32)
    B = rng.standard_normal((p, q), dtype=cp.float32) + 1j * rng.standard_normal((p, q), dtype=cp.float32)
    D = rng.standard_normal((q, s), dtype=cp.float32) + 1j * rng.standard_normal((q, s), dtype=cp.float32)

    # Explicit Kroneckers then matmul: (A⊗B)(C⊗D)
    kron_AC = cp.kron(A, B)      # shape: (m*p, n*q)
    kron_CD = cp.kron(C, D)      # shape: (n*q, r*s)
    left = kron_AC @ kron_CD     # shape: (m*p, r*s)

    # Mixed-product: (AC)⊗(BD)
    AC = A @ C                   # shape: (m, r)
    BD = B @ D                   # shape: (p, s)
    right = cp.kron(AC, BD)      # shape: (m*p, r*s)

    # Check equality (numerical)
    equal = cp.allclose(left, right, rtol=1e-4, atol=1e-5)
    max_abs_diff = cp.max(cp.abs(left - right))

    print("Shapes:")
    print(f"(A ⊗ B)(C ⊗ D): {left.shape}")
    print(f"(AC) ⊗ (BD):   {right.shape}")
    print()
    print("allclose:", bool(equal))
    print("max |difference|:", float(max_abs_diff))

    # Optional assert
    assert equal, "Mixed-product property failed (within numerical precision)."

if __name__ == "__main__":
    main()

