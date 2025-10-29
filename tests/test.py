import numpy as np

def vec_colmajor(M):
    return M.reshape(-1, order='F')

def unvec_colmajor(v, shape):
    return np.reshape(v, shape, order='F')

def kron_vec_apply_vec_trick(A, B, v, q, n):
    """
    Compute y = (A ⊗ B) @ v using the vec trick without forming the Kronecker.

    A: (m, n)
    B: (p, q)
    v: vector of length n*q = len(vec(X)) with column-major vec convention
    q, n: dimensions of X so that X has shape (q, n)

    Returns:
      y: vector of length m*p
    """
    m, nA = A.shape
    p, qB = B.shape
    assert nA == n and qB == q, "Dimension mismatch with A or B"
    assert v.size == n*q, "v must have length n*q"

    # Recreate X with shape (q, n) (column-major vec convention)
    X = unvec_colmajor(v, (q, n))          # shape (q, n)
    Y = B @ X @ A.T                        # shape (p, m)
    return vec_colmajor(Y)                 # shape (m*p,)

# --- Naive reference (forms Kronecker; for testing only) ---
def kron_vec_apply_naive(A, B, v, q, n):
    K = np.kron(A, B)                      # (m*p, n*q)
    return K @ v

# --- Demo ---
rng = np.random.default_rng(0)
m, n, p, q = 601, 601, 10, 10
A = rng.standard_normal((m, n)) + 1j*rng.standard_normal((m, n))
B = rng.standard_normal((p, q)) + 1j*rng.standard_normal((p, q))
X = rng.standard_normal((q, n)) + 1j*rng.standard_normal((q, n))  # NOTE: (q, n)
v = vec_colmajor(X)

y_trick = kron_vec_apply_vec_trick(A, B, v, q, n)
y_naive = kron_vec_apply_naive(A, B, v, q, n)

print("Match:", np.allclose(y_trick, y_naive))

