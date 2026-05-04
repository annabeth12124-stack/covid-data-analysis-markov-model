"""
Markov Disease Model Module
Implements disease spread modeling using Markov chains and linear algebra
"""

import numpy as np
from scipy.linalg import eig
import warnings
warnings.filterwarnings('ignore')


class MarkovDiseaseModel:
    """
    Implements Markov chain model for disease spread with SIR states
    """
    
    def __init__(self, state_matrix):
        """
        Initialize the Markov model with state matrix
        
        Parameters:
        -----------
        state_matrix : numpy.ndarray
            Matrix of shape (n_countries, n_timepoints * 3) containing S/I/R proportions
        """
        self.state_matrix = state_matrix
        self.transition_matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.steady_state = None
        
    def estimate_transition_matrix(self):
        """
        Estimates the 3x3 transition matrix from observed state transitions
        Using weighted probabilities instead of hard argmax
        
        Returns:
        --------
        numpy.ndarray : 3x3 transition matrix (stochastic)
        """
        # Reshape state matrix to extract transitions
        all_transitions = []  # Store (from_state_index, to_state_index, weight)
        
        for country_idx in range(self.state_matrix.shape[0]):
            country_states = self.state_matrix[country_idx].reshape(-1, 3)
            
            for t in range(len(country_states) - 1):
                current = country_states[t]
                next_state = country_states[t + 1]
                
                # Weighted approach: probability of being in each state
                # Current state distribution
                current_probs = current / (current.sum() + 1e-10)
                
                # Next state distribution
                next_probs = next_state / (next_state.sum() + 1e-10)
                
                # For each possible current state, add weighted transition
                for curr_idx in range(3):
                    if current_probs[curr_idx] > 0.1:  # Significant probability
                        # Find most likely next state
                        next_idx = np.argmax(next_probs)
                        weight = current_probs[curr_idx] * next_probs[next_idx]
                        all_transitions.append((curr_idx, next_idx, weight))
        
        # Count transitions with weights
        transition_counts = np.zeros((3, 3))
        for curr, nxt, weight in all_transitions:
            transition_counts[curr, nxt] += weight
        
        # Normalize rows to get probabilities
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        
        # Handle rows with zero counts (no observed transitions from that state)
        for i in range(3):
            if row_sums[i] == 0:
                # If no transitions observed from this state, set default probabilities
                # Default: stay in same state with high probability
                transition_counts[i] = [0.9, 0.05, 0.05]
                row_sums[i] = 1.0
        
        self.transition_matrix = transition_counts / row_sums
        
        # Ensure no NaN or Inf
        self.transition_matrix = np.nan_to_num(self.transition_matrix, nan=0.33, posinf=0.5, neginf=0)
        
        # Ensure rows sum to 1
        for i in range(3):
            row_sum = self.transition_matrix[i].sum()
            if row_sum > 0:
                self.transition_matrix[i] = self.transition_matrix[i] / row_sum
            else:
                self.transition_matrix[i] = [0.33, 0.33, 0.34]
        
        # Compute eigenvalues for later use
        self._compute_eigenvalues()
        
        return self.transition_matrix
    
    def _compute_eigenvalues(self):
        """
        Compute eigenvalues and eigenvectors of the transition matrix
        """
        if self.transition_matrix is not None:
            try:
                # Check for NaN/Inf before computing eigenvalues
                if np.any(np.isnan(self.transition_matrix)) or np.any(np.isinf(self.transition_matrix)):
                    print("Warning: Transition matrix contains NaN/Inf. Using default matrix.")
                    self.transition_matrix = np.array([
                        [0.8, 0.15, 0.05],
                        [0.1, 0.7, 0.2],
                        [0.02, 0.03, 0.95]
                    ])
                
                self.eigenvalues, self.eigenvectors = eig(self.transition_matrix)
                
                # Find steady state (eigenvector for eigenvalue closest to 1)
                idx_1 = np.argmin(np.abs(self.eigenvalues - 1.0))
                steady_eigenvector = np.real(self.eigenvectors[:, idx_1])
                
                # Ensure positive and normalize
                steady_eigenvector = np.maximum(steady_eigenvector, 0)
                if steady_eigenvector.sum() > 0:
                    self.steady_state = steady_eigenvector / steady_eigenvector.sum()
                else:
                    self.steady_state = np.array([0.33, 0.33, 0.34])
                    
            except Exception as e:
                print(f"Eigenvalue computation error: {e}")
                self.eigenvalues = np.array([1.0, 0.5, 0.2])
                self.eigenvectors = np.eye(3)
                self.steady_state = np.array([0.33, 0.33, 0.34])
    
    def calculate_r0(self):
        """
        Calculates basic reproduction number R0 from transition matrix
        
        For SIR model: R0 = transmission_rate / recovery_rate
        From transition matrix: R0 = P(S->I) / P(I->R)
        
        Returns:
        --------
        float : Estimated R0 value
        """
        if self.transition_matrix is None:
            self.estimate_transition_matrix()
        
        # R0 = probability of infection / probability of recovery
        prob_infection = self.transition_matrix[0, 1]  # S -> I
        prob_recovery = self.transition_matrix[1, 2]   # I -> R
        
        if prob_recovery > 0.01:
            r0 = prob_infection / prob_recovery
        else:
            # If no recovery observed, R0 is high
            r0 = 3.0
        
        # Cap reasonable values
        r0 = max(0.1, min(r0, 8.0))
        
        return r0
    
    def eigenvalue_analysis(self):
        """
        Performs eigenvalue analysis for system dynamics
        
        Returns:
        --------
        dict : Contains eigenvalues, dominant eigenvalue, steady state, convergence rate
        """
        if self.transition_matrix is None:
            self.estimate_transition_matrix()
        
        if self.eigenvalues is None:
            self._compute_eigenvalues()
        
        # Sort eigenvalues by magnitude
        magnitudes = np.abs(self.eigenvalues)
        sorted_indices = np.argsort(magnitudes)[::-1]
        
        dominant_idx = sorted_indices[0]
        dominant_eigenvalue = self.eigenvalues[dominant_idx]
        spectral_radius = magnitudes[dominant_idx]
        
        # Second eigenvalue determines convergence rate
        second_magnitude = 0.0
        if len(sorted_indices) > 1:
            second_magnitude = magnitudes[sorted_indices[1]]
            convergence_rate = second_magnitude
        else:
            convergence_rate = 0.0
        
        # Ensure steady state exists
        if self.steady_state is None or np.any(np.isnan(self.steady_state)):
            self.steady_state = np.array([0.33, 0.33, 0.34])
        
        return {
            'eigenvalues': self.eigenvalues,
            'dominant_eigenvalue': float(dominant_eigenvalue.real),
            'spectral_radius': float(spectral_radius),
            'second_eigenvalue_magnitude': float(second_magnitude),
            'convergence_rate': float(convergence_rate),
            'steady_state': self.steady_state
        }
    
    def predict_future_states(self, initial_state, steps):
        """
        Predicts future states using Markov chain (matrix powers)
        
        Parameters:
        -----------
        initial_state : numpy.ndarray
            Initial state vector [S, I, R]
        steps : int
            Number of steps to predict
        
        Returns:
        --------
        numpy.ndarray : Array of shape (steps, 3) with predictions
        """
        if self.transition_matrix is None:
            self.estimate_transition_matrix()
        
        predictions = []
        current_state = np.array(initial_state).copy().astype(float)
        
        # Ensure probabilities sum to 1
        current_state = current_state / (current_state.sum() + 1e-10)
        
        for _ in range(steps):
            current_state = current_state @ self.transition_matrix
            # Ensure no negative values and sum to 1
            current_state = np.maximum(current_state, 0)
            current_state = current_state / (current_state.sum() + 1e-10)
            predictions.append(current_state.copy())
        
        return np.array(predictions)
    
    def least_squares_prediction(self, observed_data, future_steps=5):
        """
        Uses least squares to fit transition matrix and predict
        
        Parameters:
        -----------
        observed_data : numpy.ndarray
            Historical observed states (n_timepoints, 3)
        future_steps : int
            Number of future steps to predict
        
        Returns:
        --------
        numpy.ndarray : Predictions for future steps
        """
        if len(observed_data) < 2:
            return self.predict_future_states(observed_data[-1] if len(observed_data) > 0 else np.array([0.9, 0.05, 0.05]), future_steps)
        
        # Build linear system: X_next = X_current * T
        X_current = observed_data[:-1]  # (n-1, 3)
        X_next = observed_data[1:]       # (n-1, 3)
        
        # Least squares solution for T
        try:
            # Using pseudo-inverse for numerical stability
            T_lsq, _, _, _ = np.linalg.lstsq(X_current, X_next, rcond=None)
            
            # Ensure no NaN/Inf
            T_lsq = np.nan_to_num(T_lsq, nan=0.33, posinf=0.5, neginf=0)
            
            # Ensure stochastic (rows sum to 1)
            for i in range(3):
                row_sum = T_lsq[i].sum()
                if row_sum > 0:
                    T_lsq[i] = T_lsq[i] / row_sum
                else:
                    T_lsq[i] = [0.33, 0.33, 0.34]
            
            # Predict using this optimized transition matrix
            current = observed_data[-1].copy()
            predictions = []
            for _ in range(future_steps):
                current = current @ T_lsq
                current = np.maximum(current, 0)
                current = current / (current.sum() + 1e-10)
                predictions.append(current.copy())
            
            return np.array(predictions)
        except Exception as e:
            print(f"Least squares failed: {e}, using regular prediction")
            # Fallback to regular transition matrix
            return self.predict_future_states(observed_data[-1], future_steps)
    
    def detect_patterns(self):
        """
        Detects patterns in disease spread using eigenvalue analysis
        
        Returns:
        --------
        dict : Pattern detection results
        """
        if self.transition_matrix is None:
            self.estimate_transition_matrix()
        
        eigen_analysis = self.eigenvalue_analysis()
        
        # Determine epidemic potential
        r0 = self.calculate_r0()
        if r0 > 2:
            epidemic_potential = "High"
        elif r0 > 1:
            epidemic_potential = "Moderate"
        else:
            epidemic_potential = "Low"
        
        # System stability (Markov chains are always stable, but convergence speed varies)
        convergence_rate = eigen_analysis['convergence_rate']
        if convergence_rate < 0.5:
            stability = "Fast convergence"
        elif convergence_rate < 0.8:
            stability = "Moderate convergence"
        else:
            stability = "Slow convergence"
        
        # Convergence speed classification
        if convergence_rate < 0.3:
            convergence_speed = "Very Fast"
        elif convergence_rate < 0.6:
            convergence_speed = "Fast"
        elif convergence_rate < 0.9:
            convergence_speed = "Moderate"
        else:
            convergence_speed = "Slow"
        
        return {
            'epidemic_potential': epidemic_potential,
            'stability': stability,
            'convergence_speed': convergence_speed,
            'r0': r0
        }
    
    def get_transition_probabilities(self):
        """
        Returns transition probabilities between states
        
        Returns:
        --------
        dict : Transition probabilities with labels
        """
        if self.transition_matrix is None:
            self.estimate_transition_matrix()
        
        return {
            'S_to_S': self.transition_matrix[0, 0],
            'S_to_I': self.transition_matrix[0, 1],
            'S_to_R': self.transition_matrix[0, 2],
            'I_to_S': self.transition_matrix[1, 0],
            'I_to_I': self.transition_matrix[1, 1],
            'I_to_R': self.transition_matrix[1, 2],
            'R_to_S': self.transition_matrix[2, 0],
            'R_to_I': self.transition_matrix[2, 1],
            'R_to_R': self.transition_matrix[2, 2]
        }