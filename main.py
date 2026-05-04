"""
Main Project File
Disease Spread Modeling Using Transition Matrices with Real COVID-19 Data
Linear Algebra Applications in Real-World Disease Analysis
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from src.data_processor import COVID19DataProcessor
from src.matrix_operations import MatrixOperations
from src.markov_model import MarkovDiseaseModel
from src.visualization import Visualizer


class DiseaseSpreadModelingProject:
    """
    Main project class implementing the complete pipeline for disease spread modeling
    """
    
    def __init__(self):
        self.processor = COVID19DataProcessor()
        self.state_matrix = None
        self.countries = None
        self.model = None
        self.ops = None
        
    def run_pipeline(self):
        """
        Execute the complete project pipeline following the guidelines
        """
        
        # STEP 1: REAL-WORLD DATA ACQUISITION
        print("="*80)
        print("STEP 1: REAL-WORLD DATA ACQUISITION")
        print("="*80)
        
        print("\nSelect countries for analysis:")
        print("Default: India, US, Brazil, UK, Germany, France, Italy, Spain, Russia, Japan")
        
        use_default = input("\nUse default countries? (y/n): ").lower()
        
        if use_default == 'y':
            countries = ['India', 'US', 'Brazil', 'UK', 'Germany', 
                        'France', 'Italy', 'Spain', 'Russia', 'Japan']
        else:
            print("Enter country names (comma-separated):")
            countries = [c.strip() for c in input().split(',')]
        
        print("\nFetching real COVID-19 data from Johns Hopkins University...")
        data = self.processor.fetch_covid19_data(countries)
        
        self.state_matrix, self.countries = self.processor.create_state_matrix(countries)
        
        stats = self.processor.get_statistics()
        print("\nData Statistics:")
        print(f"  - Countries analyzed: {stats['countries']}")
        print(f"  - Time period: {stats['date_range']}")
        print(f"  - Total data points: {stats['total_records']}")
        print(f"  - Maximum infection rate: {stats['max_infection_rate']:.2%}")
        print(f"  - Average infection rate: {stats['avg_infection_rate']:.2%}")
        
        print(f"\nMatrix Representation:")
        print(f"  - Matrix shape: {self.state_matrix.shape}")
        print(f"  - Rows (countries): {self.state_matrix.shape[0]}")
        print(f"  - Columns (time points x 3 states): {self.state_matrix.shape[1]}")
        print(f"  - Time points per country: {self.state_matrix.shape[1] // 3}")
        
        # STEP 2: INTERMEDIATE STAGE - MATRIX OPERATIONS
        print("\n" + "="*80)
        print("STEP 2: MATRIX OPERATIONS AND SIMPLIFICATION")
        print("="*80)
        
        self.ops = MatrixOperations(self.state_matrix)
        
        # 2a: RREF for matrix simplification (FIXED - handles tuple return)
        print("\n2a. Row Reduced Echelon Form (RREF) Analysis")
        rref_matrix, rref_info = self.ops.row_echelon_form()
        print(f"  - RREF applied to: {rref_info['shape_processed'][0]}×{rref_info['shape_processed'][1]} matrix")
        print(f"  - Demo mode: {rref_info['demo_mode']}")
        print(f"  - Matrix rank: {rref_info['rank']}")
        print(f"  - Pivot columns: {rref_info['pivot_columns']}")
        print(f"  - Data redundancy: {self.state_matrix.shape[1] - rref_info['rank']} dimensions")
        
        # 2b: Orthogonal Basis Formation
        print("\n2b. Orthogonal Basis Formation (QR Decomposition)")
        basis_result = self.ops.find_basis()
        print(f"  - Basis dimension: {basis_result['rank']}")
        print(f"  - Orthogonal basis shape: {basis_result['orthogonal_basis'].shape}")
        print(f"  - Redundancy removed: {basis_result['redundancy_removed']} columns")
        print(f"  - Data can be represented with {basis_result['rank']} basis vectors")
        
        # 2c: Matrix Simplification via SVD
        print("\n2c. Matrix Simplification (SVD Analysis)")
        simplification = self.ops.matrix_simplification(variance_threshold=0.95)
        print(f"  - Optimal rank for 95% variance: {simplification['optimal_rank']}")
        print(f"  - Compression ratio: {simplification['compression_ratio']:.2%}")
        print(f"  - Variance retained: {simplification['variance_retained']:.2%}")
        print(f"  - Reconstruction error: {simplification['reconstruction_error']:.4f}")
        print(f"  - Top 3 singular values: {simplification['singular_values'][:3]}")
        print(f"  - Explained variance by first component: {simplification['explained_variance'][0]:.2%}")
        
        # STEP 3: FINAL APPLICATION OUTPUT
        print("\n" + "="*80)
        print("STEP 3: DISEASE MODELING AND PREDICTIONS")
        print("="*80)
        
        self.model = MarkovDiseaseModel(self.state_matrix)
        
        # 3a: Transition Matrix from Real Data
        print("\n3a. Transition Matrix Estimation")
        transition_matrix = self.model.estimate_transition_matrix()
        print(f"  Transition Matrix (S, I, R):")
        print(f"    S -> S: {transition_matrix[0,0]:.3f}")
        print(f"    S -> I: {transition_matrix[0,1]:.3f}")
        print(f"    S -> R: {transition_matrix[0,2]:.3f}")
        print(f"    I -> S: {transition_matrix[1,0]:.3f}")
        print(f"    I -> I: {transition_matrix[1,1]:.3f}")
        print(f"    I -> R: {transition_matrix[1,2]:.3f}")
        print(f"    R -> S: {transition_matrix[2,0]:.3f}")
        print(f"    R -> I: {transition_matrix[2,1]:.3f}")
        print(f"    R -> R: {transition_matrix[2,2]:.3f}")
        
        # 3b: R0 Calculation
        print("\n3b. Basic Reproduction Number (R0)")
        r0 = self.model.calculate_r0()
        print(f"  - Estimated R0: {r0:.2f}")
        if r0 > 1:
            print(f"  - Interpretation: Disease is spreading (R0 > 1)")
        else:
            print(f"  - Interpretation: Disease is declining (R0 < 1)")
        
        # 3c: Eigenvalue Analysis
        print("\n3c. Eigenvalue/Eigenvector Analysis")
        eigen_analysis = self.model.eigenvalue_analysis()
        print(f"  - Dominant eigenvalue: {eigen_analysis['dominant_eigenvalue']:.4f}")
        print(f"  - Spectral radius: {eigen_analysis['spectral_radius']:.4f}")
        print(f"  - Second eigenvalue magnitude: {eigen_analysis.get('second_eigenvalue_magnitude', 'N/A')}")
        print(f"  - Convergence rate: {eigen_analysis['convergence_rate']:.4f}")
        
        stability_status = "Stable (Markov chain)" if eigen_analysis['spectral_radius'] <= 1 else "Unstable"
        print(f"  - System stability: {stability_status}")
        print(f"  - Steady state from eigenvector: S={eigen_analysis['steady_state'][0]:.3f}, "
              f"I={eigen_analysis['steady_state'][1]:.3f}, R={eigen_analysis['steady_state'][2]:.3f}")
        
        # 3d: Pattern Detection
        print("\n3d. Pattern Detection and Analysis")
        patterns = self.model.detect_patterns()
        print(f"  - Epidemic Potential: {patterns['epidemic_potential']}")
        print(f"  - System Stability: {patterns['stability']}")
        print(f"  - Convergence Speed: {patterns['convergence_speed']}")
        
        # 3e: Future Predictions (Markov Chain)
        print("\n3e. Future State Predictions (Next 8 weeks)")
        # Get the last state of the first country
        current_state = self.state_matrix[0].reshape(-1, 3)[-1]
        future_steps = 8
        predictions = self.model.predict_future_states(current_state, future_steps)
        
        print(f"\n  Current state ({self.countries[0]}):")
        print(f"    Susceptible: {current_state[0]:.2%}")
        print(f"    Infected: {current_state[1]:.2%}")
        print(f"    Recovered: {current_state[2]:.2%}")
        
        print(f"\n  Markov Chain Predictions for next {future_steps} weeks:")
        for i in range(min(5, future_steps)):
            print(f"    Week {i+1}: S={predictions[i,0]:.2%}, I={predictions[i,1]:.2%}, R={predictions[i,2]:.2%}")
        
        # 3f: Least Squares Optimization
        print("\n3f. Least Squares Optimization for Predictions")
        observed_data = self.state_matrix[0].reshape(-1, 3)
        lsq_predictions = self.model.least_squares_prediction(observed_data, future_steps=5)
        
        print(f"  Least squares predictions (optimized using historical data):")
        for i in range(min(3, len(lsq_predictions))):
            print(f"    Week {i+1}: S={lsq_predictions[i,0]:.2%}, I={lsq_predictions[i,1]:.2%}, R={lsq_predictions[i,2]:.2%}")
        
        # 3g: Long-term Analysis
        print("\n3g. Long-term Steady State Analysis")
        print(f"  Long-term distribution (after convergence):")
        print(f"    Susceptible: {eigen_analysis['steady_state'][0]:.2%}")
        print(f"    Infected: {eigen_analysis['steady_state'][1]:.2%}")
        print(f"    Recovered: {eigen_analysis['steady_state'][2]:.2%}")
        
        # Check if disease persists
        if eigen_analysis['steady_state'][1] > 0.01:
            print(f"  ⚠️ Disease becomes endemic ({(eigen_analysis['steady_state'][1]*100):.1f}% population infected long-term)")
        else:
            print(f"  ✅ Disease dies out in long-term")
        
        # VISUALIZATION
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        visualizer = Visualizer(self.processor, self.model, self.countries)
        visualizer.plot_covid_trends(save_path='covid19_trends.png')
        visualizer.plot_transition_analysis(save_path='transition_analysis.png')
        visualizer.plot_predictions(predictions, future_steps, save_path='disease_predictions.png')
        visualizer.plot_svd_analysis(save_path='svd_analysis.png')
        
        print("\nVisualizations saved:")
        print("  - covid19_trends.png")
        print("  - transition_analysis.png")
        print("  - disease_predictions.png")
        print("  - svd_analysis.png")
        
        # FINAL SUMMARY
        print("\n" + "="*80)
        print("PROJECT SUMMARY")
        print("="*80)
        
        print("\nLinear Algebra Concepts Applied to Real COVID-19 Data:")
        print(f"  1. Matrix Representation - Organized pandemic data from {len(self.countries)} countries")
        print(f"  2. RREF - Applied to {rref_info['shape_processed'][0]}×{rref_info['shape_processed'][1]} demo submatrix")
        print(f"  3. QR Decomposition - Found orthogonal basis with {basis_result['rank']} vectors")
        print(f"  4. SVD - Compressed data by {simplification['compression_ratio']:.1%} while retaining {simplification['variance_retained']:.1%} variance")
        print(f"  5. Transition Matrix - Modeled COVID-19 transmission probabilities")
        print(f"  6. Eigenvalue Analysis - Determined pandemic dynamics and stability")
        print(f"  7. Markov Chain - Predicted future COVID-19 trends")
        print(f"  8. Least Squares - Optimized predictions using historical data")
        
        print("\nMeaningful Outcomes Achieved with Real Data:")
        print(f"  - Pattern Detection: {patterns['epidemic_potential']} epidemic potential")
        print(f"  - Future Predictions: Forecasted COVID-19 spread for {future_steps} weeks")
        print(f"  - Data Compression: Reduced dimensionality by {simplification['compression_ratio']:.1%}")
        print(f"  - Noise Reduction: Reconstruction error = {simplification['reconstruction_error']:.4f}")
        print(f"  - System Analysis: Found long-term steady state distribution")
        print(f"  - Country Comparison: Analyzed {len(self.countries)} countries simultaneously")
        print(f"  - R0 Estimation: Basic reproduction number = {r0:.2f}")
        
        # Export results
        self._export_results(self.model, predictions, patterns, r0, simplification, eigen_analysis)
        
        return {
            'processor': self.processor,
            'model': self.model,
            'predictions': predictions,
            'patterns': patterns,
            'countries': self.countries,
            'r0': r0,
            'simplification': simplification,
            'eigen_analysis': eigen_analysis
        }
    
    def _export_results(self, model, predictions, patterns, r0, simplification, eigen_analysis):
        """
        Export analysis results to CSV
        """
        results = {
            'Metric': [
                'Epidemic Potential', 
                'System Stability', 
                'Convergence Speed',
                'R0 Value',
                'Dominant Eigenvalue', 
                'Spectral Radius',
                'Steady State S', 
                'Steady State I', 
                'Steady State R',
                'Optimal Rank (95% variance)',
                'Compression Ratio',
                'Reconstruction Error'
            ],
            'Value': [
                patterns['epidemic_potential'], 
                patterns['stability'], 
                patterns['convergence_speed'],
                f'{r0:.4f}',
                f'{eigen_analysis["dominant_eigenvalue"]:.6f}',
                f'{eigen_analysis["spectral_radius"]:.6f}',
                f'{eigen_analysis["steady_state"][0]:.6f}',
                f'{eigen_analysis["steady_state"][1]:.6f}',
                f'{eigen_analysis["steady_state"][2]:.6f}',
                simplification['optimal_rank'],
                f'{simplification["compression_ratio"]:.4f}',
                f'{simplification["reconstruction_error"]:.6f}'
            ]
        }
        
        results_df = pd.DataFrame(results)
        results_df.to_csv('covid19_analysis_results.csv', index=False)
        print("\n✅ Results exported to 'covid19_analysis_results.csv'")
        
        # Also export transition matrix
        if model.transition_matrix is not None:
            trans_df = pd.DataFrame(
                model.transition_matrix,
                index=['From_S', 'From_I', 'From_R'],
                columns=['To_S', 'To_I', 'To_R']
            )
            trans_df.to_csv('transition_matrix.csv')
            print("✅ Transition matrix exported to 'transition_matrix.csv'")


if __name__ == "__main__":
    print("="*80)
    print("DISEASE SPREAD MODELING USING TRANSITION MATRICES")
    print("Linear Algebra Applications in Real-World Disease Analysis")
    print("="*80)
    print("\nThis project uses real COVID-19 data from Johns Hopkins University")
    print("and applies linear algebra concepts for prediction and analysis.\n")
    print("Linear Algebra Concepts Demonstrated:")
    print("  • Matrix Representation & RREF")
    print("  • QR Decomposition (Orthogonal Basis)")
    print("  • SVD (Dimensionality Reduction & Noise Removal)")
    print("  • Eigenvalues/Eigenvectors (Steady State Analysis)")
    print("  • Markov Chains (Prediction)")
    print("  • Least Squares (Optimization)")
    print("="*80)
    
    project = DiseaseSpreadModelingProject()
    results = project.run_pipeline()
    
    print("\n" + "="*80)
    print("✅ PROJECT COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nCheck the following files generated:")
    print("  • covid19_trends.png")
    print("  • transition_analysis.png")
    print("  • disease_predictions.png")
    print("  • svd_analysis.png")
    print("  • covid19_analysis_results.csv")
    print("  • transition_matrix.csv")