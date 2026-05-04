"""
Matrix Operations Module
Implements linear algebra operations for matrix analysis

IMPORTANT CONTEXT FOR THIS PROJECT:
- RREF is mathematically correct but produces trivial output on large (countries × many time steps) matrices.
- For DEMO purposes, RREF is applied here to a small 3×3 submatrix so the output is interpretable.
- SVD and QR are the PRIMARY tools for dimensionality reduction and basis finding in this disease spread model.
"""

import numpy as np
from scipy.linalg import svd, qr
import pandas as pd


class MatrixOperations:
    """
    Implements various linear algebra operations for matrix analysis
    """
    
    def __init__(self, matrix):
        """
        Initialize with input matrix
        
        Parameters:
        -----------
        matrix : numpy.ndarray
            Input matrix for analysis
        """
        self.matrix = matrix
        self.reduced_form = None
        self.basis = None
        
    def row_echelon_form(self, use_demo_submatrix=True, demo_size=3):
        """
        Computes Row Reduced Echelon Form (RREF).
        
        NOTE FOR DISEASE SPREAD MODEL:
        - On the full state matrix (countries × timepoints*3), RREF typically produces an identity-like matrix
          because most columns are linearly independent. This is mathematically correct but NOT insightful.
        - For DEMONSTRATION of the RREF concept, this method can optionally work on a small interpretable submatrix.
        
        Parameters:
        -----------
        use_demo_submatrix : bool, default=True
            If True, applies RREF to a small square submatrix (first demo_size rows/cols)
            so the output is clean and interpretable for teaching.
            If False, applies RREF to the full matrix (may be large and less meaningful).
        demo_size : int, default=3
            Size of the demo submatrix (only used if use_demo_submatrix=True)
        
        Returns:
        --------
        numpy.ndarray : RREF of the selected matrix
        dict : Additional info about the operation
        """
        if use_demo_submatrix and self.matrix.shape[0] >= demo_size and self.matrix.shape[1] >= demo_size:
            # Take a small square submatrix for clean RREF demonstration
            A_demo = self.matrix[:demo_size, :demo_size].copy().astype(float)
            print(f"RREF applied to {demo_size}×{demo_size} demo submatrix (rows 0-{demo_size-1}, cols 0-{demo_size-1})")
            print("  Reason: Full state matrix RREF is identity-like and not interpretable for disease spread.")
            A = A_demo
            original_shape = self.matrix.shape
        else:
            A = self.matrix.copy().astype(float)
            original_shape = A.shape
            print(f"RREF applied to full {original_shape[0]}×{original_shape[1]} matrix")
            print("  Note: Result may be identity-like with many pivot columns")
        
        rows, cols = A.shape
        lead = 0
        
        for r in range(rows):
            if lead >= cols:
                break
            
            i = r
            while i < rows and abs(A[i, lead]) < 1e-10:
                i += 1
            
            if i < rows:
                A[[r, i]] = A[[i, r]]
                pivot = A[r, lead]
                
                if abs(pivot) > 1e-10:
                    A[r] = A[r] / pivot
                
                for j in range(rows):
                    if j != r:
                        A[j] = A[j] - A[j, lead] * A[r]
                
                lead += 1
        
        self.reduced_form = A
        
        # Count pivot columns for interpretation
        pivots = [j for j in range(cols) if any(abs(A[i, j]) > 1e-8 for i in range(rows))]
        
        return A, {
            'shape_processed': A.shape,
            'original_shape': original_shape,
            'pivot_columns': pivots,
            'rank': len(pivots),
            'demo_mode': use_demo_submatrix and self.matrix.shape[0] >= demo_size and self.matrix.shape[1] >= demo_size
        }
    
    def find_basis(self):
        """
        Finds basis and orthogonal basis using QR decomposition.
        
        WHY THIS IS NEEDED:
        - Identifies linearly independent columns in the state matrix.
        - Removes redundancy across time points.
        - Orthogonal basis is useful for projection later.
        
        Returns:
        --------
        dict : Contains orthogonal basis, rank, and shape information
        """
        Q, R = qr(self.matrix, mode='economic')
        
        rank = np.sum(np.abs(np.diag(R)) > 1e-10)
        
        self.basis = Q[:, :rank]
        
        return {
            'orthogonal_basis': self.basis,
            'rank': rank,
            'shape': self.basis.shape,
            'redundancy_removed': self.matrix.shape[1] - rank
        }
    
    def matrix_simplification(self, variance_threshold=0.95):
        """
        Performs matrix simplification using Singular Value Decomposition (SVD).
        
        WHY THIS IS NEEDED:
        - SVD finds optimal low-rank approximation of the disease state matrix.
        - Captures dominant patterns (e.g., first wave, second wave) as singular vectors.
        - Removes noise by keeping only high-variance components.
        
        Parameters:
        -----------
        variance_threshold : float, default=0.95
            Fraction of variance to retain (between 0 and 1)
        
        Returns:
        --------
        dict : Contains simplified matrix, singular values, and compression metrics
        """
        U, s, Vt = svd(self.matrix, full_matrices=False)
        
        singular_values = s
        
        energy = np.cumsum(s**2) / np.sum(s**2)
        optimal_rank = np.argmax(energy >= variance_threshold) + 1
        
        simplified_matrix = U[:, :optimal_rank] @ np.diag(s[:optimal_rank]) @ Vt[:optimal_rank, :]
        
        # Calculate reconstruction error
        reconstruction_error = np.linalg.norm(self.matrix - simplified_matrix, 'fro') / np.linalg.norm(self.matrix, 'fro')
        
        return {
            'simplified_matrix': simplified_matrix,
            'singular_values': singular_values,
            'optimal_rank': optimal_rank,
            'compression_ratio': 1 - (optimal_rank / len(s)),
            'variance_retained': energy[optimal_rank - 1],
            'reconstruction_error': reconstruction_error,
            'U_matrix': U,
            'Vt_matrix': Vt,
            'explained_variance': s**2 / np.sum(s**2)
        }
    
    def get_matrix_properties(self):
        """
        Returns fundamental matrix properties
        
        Returns:
        --------
        dict : Matrix properties including rank, norm, condition number
        """
        rank = np.linalg.matrix_rank(self.matrix)
        norm = np.linalg.norm(self.matrix)
        
        if self.matrix.shape[0] == self.matrix.shape[1]:
            condition_number = np.linalg.cond(self.matrix)
        else:
            condition_number = 'N/A (non-square matrix)'
        
        return {
            'shape': self.matrix.shape,
            'rank': rank,
            'nullity': self.matrix.shape[1] - rank,
            'norm': norm,
            'condition_number': condition_number
        }