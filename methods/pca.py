import numpy as np

class pca:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def QR(self, A):
        A = np.array(A, dtype=float)
        m, n = A.shape
        
        Q = np.zeros((m, n))
        R = np.zeros((n, n))
        
        for j in range(n):
            v = A[:, j].copy()
            
            for i in range(j):
                R[i, j] = np.dot(Q[:, i], A[:, j])
                v -= R[i, j] * Q[:, i]
            
            norm_v = np.linalg.norm(v)
                
            R[j, j] = norm_v
            Q[:, j] = v / norm_v
        
        return Q, R


    def fit_transform(self, X):
        X = np.array(X, dtype=float)
        X_normalized = X

        for i in range(X.shape[1]):
            x_i = X[:, i]
            x_i = x_i - np.mean(x_i)
            std = np.sqrt(np.sum(x_i ** 2) / len(x_i))
            X_normalized[:, i] = x_i / std
        
        cov_matrix = (X_normalized.T @ X_normalized) / len(X_normalized)

        Q_total = np.eye(cov_matrix.shape[0])
        A = cov_matrix.copy()
        for _ in range(10):
            Q, R = self.QR(A)
            A = R @ Q
            Q_total = Q_total @ Q

        vals = np.diag(A)
        vecs = Q_total
        
        sorted_indices = np.argsort(vals)[::-1]
        sorted_vecs = vecs[:, sorted_indices]

        W = sorted_vecs[:, :self.n_components]

        return X @ W