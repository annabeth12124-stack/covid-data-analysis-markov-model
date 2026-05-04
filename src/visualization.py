import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class Visualizer:
    """
    Visualization utilities for the disease modeling project
    
    All plots are designed to demonstrate linear algebra concepts:
    - Transition matrix heatmap → matrix representation
    - Eigenvalue spectrum → stability analysis
    - SVD plots → dimensionality reduction
    - Predictions → model output
    """
    
    def __init__(self, processor, model, countries):
        """
        Initialize visualizer with data and model
        """
        self.processor = processor
        self.model = model
        self.countries = countries
        self.data = processor.raw_data
        self._setup_style()
    
    def _setup_style(self):
        """Configure plotting style"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_covid_trends(self, save_path=None):
        """
        Plot COVID-19 trends for selected countries
        
        WHAT THIS SHOWS:
        - Infection rates over time (raw data)
        - Current infection rates by country
        - Transition matrix as heatmap (linear algebra concept)
        - Steady state distribution from eigenvectors
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Infection rates over time
        ax1 = axes[0, 0]
        for country in self.countries[:5]:
            country_data = self.data[self.data['country'] == country].copy()
            country_data['date'] = pd.to_datetime(country_data['date'])
            ax1.plot(country_data['date'], country_data['infected'] * 100,
                    label=country, linewidth=2, alpha=0.7)
        
        ax1.set_title('Infection Rates Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Infection Rate (%)')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: Current infection rates comparison
        ax2 = axes[0, 1]
        current_infection = []
        countries_plot = []
        
        for country in self.countries:
            country_data = self.data[self.data['country'] == country].sort_values('date')
            if len(country_data) > 0:
                current_infection.append(country_data.iloc[-1]['infected'] * 100)
                countries_plot.append(country)
        
        bars = ax2.barh(countries_plot, current_infection, color='coral')
        ax2.set_title('Current Infection Rates by Country', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Infection Rate (%)')
        ax2.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, value) in enumerate(zip(bars, current_infection)):
            ax2.text(value, bar.get_y() + bar.get_height()/2, f' {value:.1f}%',
                    va='center', fontsize=9)
        
        # Plot 3: Transition matrix heatmap
        ax3 = axes[1, 0]
        if self.model.transition_matrix is not None:
            sns.heatmap(self.model.transition_matrix,
                       annot=True, fmt='.3f', cmap='RdYlGn_r',
                       xticklabels=['S', 'I', 'R'],
                       yticklabels=['S', 'I', 'R'],
                       ax=ax3, cbar_kws={'label': 'Probability'})
            ax3.set_title('State Transition Matrix (Linear Algebra: Matrix Representation)', 
                         fontsize=12, fontweight='bold')
            ax3.set_xlabel('Next State')
            ax3.set_ylabel('Current State')
        
        # Plot 4: Steady state distribution
        ax4 = axes[1, 1]
        if self.model.steady_state is not None:
            states = ['Susceptible', 'Infected', 'Recovered']
            colors = ['#2ecc71', '#e74c3c', '#3498db']
            bars = ax4.bar(states, self.model.steady_state * 100, color=colors)
            ax4.set_title('Long-term Steady State Distribution (From Eigenvector with λ=1)', 
                         fontsize=12, fontweight='bold')
            ax4.set_ylabel('Percentage (%)')
            ax4.set_ylim(0, 100)
            
            for bar, value in zip(bars, self.model.steady_state * 100):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.1f}%', ha='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_transition_analysis(self, save_path=None):
        """
        Plot eigenvalue and transition analysis
        
        WHAT THIS SHOWS:
        - Eigenvalue spectrum with unit circle → stability of the Markov chain
        - Transition probabilities bar chart → matrix entries
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Eigenvalue spectrum
        ax1 = axes[0]
        if self.model.eigenvalues is not None:
            eigenvalues = self.model.eigenvalues
            ax1.scatter(eigenvalues.real, eigenvalues.imag, s=100, alpha=0.6, 
                       c='blue', edgecolors='black')
            ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            
            # Unit circle (|λ| = 1 is stability boundary for Markov chains)
            circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--', alpha=0.5)
            ax1.add_patch(circle)
            
            ax1.set_title('Eigenvalue Spectrum - System Dynamics\n(λ=1 gives steady state)', 
                         fontsize=12, fontweight='bold')
            ax1.set_xlabel('Real Part')
            ax1.set_ylabel('Imaginary Part')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(-1.2, 1.2)
            ax1.set_ylim(-1.2, 1.2)
        
        # Plot 2: Transition probabilities
        ax2 = axes[1]
        if self.model.transition_matrix is not None:
            transitions = ['S→S', 'S→I', 'S→R', 'I→S', 'I→I', 'I→R', 'R→S', 'R→I', 'R→R']
            probabilities = self.model.transition_matrix.flatten()
            
            colors = ['green' if t in ['S→I', 'I→R'] else 'gray' for t in transitions]
            ax2.bar(transitions, probabilities, color=colors, alpha=0.7, edgecolor='black')
            ax2.set_title('State Transition Probabilities (Matrix Entries)', 
                         fontsize=12, fontweight='bold')
            ax2.set_xlabel('State Transitions')
            ax2.set_ylabel('Probability')
            ax2.set_xticklabels(transitions, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_predictions(self, predictions, future_steps, save_path=None):
        """
        Plot predictions for disease spread
        
        WHAT THIS SHOWS:
        - Historical data vs. model predictions
        - Uncertainty grows with prediction horizon
        
        IMPORTANT NOTE ON CONFIDENCE INTERVAL:
        The confidence interval shown here uses a DEMO HEURISTIC: width = 0.1 * (1 - e^(-t)).
        This is NOT derived from the model's actual uncertainty propagation.
        In a full model, uncertainty would come from:
        - Eigenvalue sensitivity (condition number)
        - Residual variance after least squares fitting
        - Bootstrap resampling of transition matrix
        
        For DEMO PURPOSES, this heuristic visually shows that uncertainty increases with time.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        first_country = self.countries[0]
        country_data = self.data[self.data['country'] == first_country].sort_values('date').copy()
        country_data['date'] = pd.to_datetime(country_data['date'])
        
        # Get historical data (last 20 points)
        historical = country_data.tail(20)
        historical_dates = historical['date']
        
        # Plot historical data
        ax.plot(historical_dates, historical['infected'] * 100,
               label='Historical Infection Rate', linewidth=2, color='blue', 
               marker='o', markersize=4)
        
        # Generate future dates
        last_date = historical_dates.iloc[-1]
        future_dates = pd.date_range(last_date, periods=future_steps + 1, freq='W')[1:]
        
        # Plot predictions
        ax.plot(future_dates, predictions[:, 1] * 100,
               label='Predicted Infection Rate', linewidth=2,
               linestyle='--', marker='s', markersize=6, color='red')
        
        # DEMO HEURISTIC for confidence interval
        # Explanation for viva: "We used a simple increasing function 1-e^(-t) to visually
        # demonstrate that prediction uncertainty grows with time. A real model would compute
        # this from eigenvalue sensitivity or bootstrap."
        uncertainty = 0.1 * (1 - np.exp(-np.arange(future_steps)))
        upper_bound = predictions[:, 1] * (1 + uncertainty) * 100
        lower_bound = predictions[:, 1] * (1 - uncertainty) * 100
        ax.fill_between(future_dates, lower_bound, upper_bound,
                        alpha=0.2, color='red', label='Prediction Interval (Demo Heuristic)')
        
        ax.set_title(f'Disease Spread Predictions for {first_country}\n(Uncertainty grows with time)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Infection Rate (%)')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format dates on x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_svd_analysis(self, save_path=None):
        """
        Plot SVD analysis results
        
        WHAT THIS SHOWS:
        - Singular values decay → low-rank structure exists
        - Cumulative explained variance → how many components to keep
        - Used for dimensionality reduction and noise removal
        
        LINEAR ALGEBRA CONCEPT: SVD decomposes data into U·Σ·Vᵀ
        - U: Left singular vectors (patterns across countries)
        - Σ: Singular values (importance of each pattern)
        - Vᵀ: Right singular vectors (patterns across time)
        """
        from scipy.linalg import svd
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Perform SVD
        U, s, Vt = svd(self.model.state_matrix, full_matrices=False)
        
        # Plot 1: Singular values
        ax1 = axes[0]
        ax1.plot(range(1, len(s) + 1), s, 'bo-', linewidth=2, markersize=6)
        ax1.set_title('Singular Values Spectrum\n(Decay indicates low-rank structure)', 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Singular Value')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Cumulative explained variance
        ax2 = axes[1]
        explained_variance = s**2 / np.sum(s**2)
        cumulative_variance = np.cumsum(explained_variance)
        
        ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100,
                'r-', linewidth=2, marker='o', markersize=6)
        ax2.axhline(y=95, color='g', linestyle='--', label='95% Variance Threshold')
        ax2.set_title('Cumulative Explained Variance\n(For dimensionality reduction)', 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('Number of Components (Rank)')
        ax2.set_ylabel('Cumulative Explained Variance (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_svd_reconstruction(self, rank=None, save_path=None):
        """
        Plot original vs. SVD-reconstructed data for different ranks
        
        PARAMETERS:
        -----------
        rank : int or None
            If None, uses rank that captures 95% variance
        """
        from scipy.linalg import svd
        
        U, s, Vt = svd(self.model.state_matrix, full_matrices=False)
        
        # Determine rank
        if rank is None:
            explained_variance = s**2 / np.sum(s**2)
            cumulative = np.cumsum(explained_variance)
            rank = np.argmax(cumulative >= 0.95) + 1
        
        # Reconstruct
        reconstructed = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]
        
        # Plot first country's infection curve
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Extract infection values (every 3rd column starting from index 1)
        original_infection = self.model.state_matrix[0, 1::3]
        reconstructed_infection = reconstructed[0, 1::3]
        
        time_points = range(len(original_infection))
        
        ax.plot(time_points, original_infection * 100, 'b-', linewidth=2, label='Original', alpha=0.7)
        ax.plot(time_points, reconstructed_infection * 100, 'r--', linewidth=2, label=f'Reconstructed (rank={rank})', alpha=0.7)
        
        ax.set_title(f'SVD Reconstruction: Original vs. Low-Rank Approximation\n(rank={rank}, captures {cumulative[rank-1]*100:.1f}% variance)',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Infection Rate (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()