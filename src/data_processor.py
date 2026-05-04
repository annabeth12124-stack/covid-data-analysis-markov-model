"""
COVID-19 Data Processor Module
Handles fetching and processing real-world disease data from JHU CSSE
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class COVID19DataProcessor:
    """
    Processes real COVID-19 data from Johns Hopkins University
    Converts to SIR model format for linear algebra analysis
    """
    
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        self.population_data = {}
        
    def load_population_data(self):
        """
        Loads population data for normalization
        """
        self.population_data = {
            'US': 331000000,
            'India': 1380000000,
            'Brazil': 213000000,
            'UK': 67800000,
            'Germany': 83200000,
            'France': 67400000,
            'Italy': 60400000,
            'Spain': 47400000,
            'Russia': 146000000,
            'Japan': 126000000,
            'Australia': 25700000,
            'Canada': 38200000,
            'South Africa': 59300000,
            'Mexico': 128000000,
            'Indonesia': 273000000
        }
    
    def fetch_covid19_data(self, countries=None, start_date=None, end_date=None):
        """
        Fetches real COVID-19 data from JHU CSSE repository
        
        Parameters:
        -----------
        countries : list, optional
            List of country names to fetch
        start_date : str, optional
            Start date for data (format: 'YYYY-MM-DD')
        end_date : str, optional
            End date for data (format: 'YYYY-MM-DD')
        
        Returns:
        --------
        pandas.DataFrame : Processed COVID-19 data in SIR format
        """
        print("Fetching real COVID-19 data from Johns Hopkins University...")
        
        confirmed_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
        deaths_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
        recovered_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"
        
        try:
            confirmed_df = pd.read_csv(confirmed_url)
            deaths_df = pd.read_csv(deaths_url)
            recovered_df = pd.read_csv(recovered_url)
            
            print(f"Data fetched successfully:")
            print(f"  - Confirmed cases: {confirmed_df.shape}")
            print(f"  - Deaths: {deaths_df.shape}")
            print(f"  - Recovered: {recovered_df.shape}")
            
            if countries is None:
                countries = ['India', 'US', 'Brazil', 'UK', 'Germany', 
                           'France', 'Italy', 'Spain', 'Russia', 'Japan']
            
            self.load_population_data()
            self.raw_data = self._process_country_data(
                confirmed_df, deaths_df, recovered_df, countries
            )
            
            if start_date or end_date:
                self.raw_data = self._filter_dates(self.raw_data, start_date, end_date)
            
            return self.raw_data
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            print("Using fallback demo data (NOT epidemiologically realistic — for demonstration only)")
            return self._create_demo_fallback_data()
    
    def _process_country_data(self, confirmed_df, deaths_df, recovered_df, countries):
        """
        Process data for specified countries
        """
        processed = []
        
        for country in countries:
            country_confirmed = confirmed_df[confirmed_df['Country/Region'] == country]
            country_deaths = deaths_df[deaths_df['Country/Region'] == country]
            country_recovered = recovered_df[recovered_df['Country/Region'] == country]
            
            if len(country_confirmed) == 0:
                print(f"No data found for {country}")
                continue
            
            dates = confirmed_df.columns[4:]
            
            confirmed_cumulative = country_confirmed.iloc[:, 4:].values.flatten()
            deaths_cumulative = country_deaths.iloc[:, 4:].values.flatten() if len(country_deaths) > 0 else np.zeros(len(confirmed_cumulative))
            recovered_cumulative = country_recovered.iloc[:, 4:].values.flatten() if len(country_recovered) > 0 else np.zeros(len(confirmed_cumulative))
            
            active_cases = confirmed_cumulative - deaths_cumulative - recovered_cumulative
            
            population = self.population_data.get(country, 10000000)
            
            for i in range(len(dates)):
                if i > 0:
                    susceptible = max(0, population - confirmed_cumulative[i])
                    infected = active_cases[i]
                    recovered = recovered_cumulative[i] + deaths_cumulative[i]
                    
                    total = susceptible + infected + recovered
                    if total > 0:
                        processed.append({
                            'country': country,
                            'date': dates[i],
                            'susceptible': susceptible / total,
                            'infected': infected / total,
                            'recovered': recovered / total,
                            'confirmed_cases': confirmed_cumulative[i],
                            'active_cases': active_cases[i],
                            'deaths': deaths_cumulative[i]
                        })
        
        df = pd.DataFrame(processed)
        print(f"Processed data for {len(df['country'].unique())} countries")
        print(f"  - Time range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
        print(f"  - Total records: {len(df)}")
        
        return df
    
    def _filter_dates(self, df, start_date=None, end_date=None):
        """
        Filter dataframe by date range
        """
        df['date'] = pd.to_datetime(df['date'])
        
        if start_date:
            start = pd.to_datetime(start_date)
            df = df[df['date'] >= start]
        
        if end_date:
            end = pd.to_datetime(end_date)
            df = df[df['date'] <= end]
        
        return df
    
    def _create_demo_fallback_data(self):
        """
        Creates DEMO-ONLY fallback data.
        
        WARNING: This does NOT satisfy SIR differential equations.
        It generates smooth S/I/R curves for LINEAR ALGEBRA DEMONSTRATION only
        (matrix representation, eigenvalue analysis, projections).
        Do NOT use for actual epidemiological prediction.
        """
        print("Creating demo fallback data with smooth S/I/R proportions...")
        print("WARNING: This is for linear algebra demonstration only, not realistic disease dynamics.")
        
        countries = ['India', 'US', 'Brazil', 'UK', 'Germany', 'France']
        dates = pd.date_range('2020-03-01', periods=100, freq='D')
        
        data = []
        for country in countries:
            # Generate smooth SIR-like curves that sum to 1
            t = np.linspace(0, 1, 100)
            
            # Susceptible starts high, decays
            susceptible = 0.95 * np.exp(-3 * t)
            # Infected rises then falls
            infected = 0.7 * t * np.exp(-4 * (t - 0.3)**2)
            # Recovered increases monotonically
            recovered = 1 - susceptible - infected
            
            # Ensure no negative values
            susceptible = np.maximum(susceptible, 0)
            infected = np.maximum(infected, 0)
            recovered = np.maximum(recovered, 0)
            
            # Renormalize to sum to 1
            total = susceptible + infected + recovered
            susceptible = susceptible / total
            infected = infected / total
            recovered = recovered / total
            
            for i, date in enumerate(dates):
                data.append({
                    'country': country,
                    'date': date,
                    'susceptible': susceptible[i],
                    'infected': infected[i],
                    'recovered': recovered[i]
                })
        
        return pd.DataFrame(data)
    
    def create_state_matrix(self, countries=None, smooth=True):
        """
        Creates state matrix from processed COVID-19 data
        
        Parameters:
        -----------
        countries : list, optional
            List of countries to include
        smooth : bool, default=True
            Apply smoothing to reduce noise
        
        Returns:
        --------
        tuple : (state_matrix, country_names)
            state_matrix: numpy array of shape (n_countries, n_timepoints * 3)
            country_names: list of country names
        """
        if self.raw_data is None:
            self.fetch_covid19_data(countries)
        
        if countries is None:
            countries = self.raw_data['country'].unique()
        
        state_matrix = []
        country_names = []
        
        for country in countries:
            country_data = self.raw_data[self.raw_data['country'] == country].sort_values('date')
            
            if len(country_data) > 0:
                country_names.append(country)
                state_vector = []
                
                for _, row in country_data.iterrows():
                    state_vector.extend([
                        row['susceptible'],
                        row['infected'],
                        row['recovered']
                    ])
                
                if smooth:
                    state_vector = self._smooth_data(state_vector)
                
                state_matrix.append(state_vector)
        
        return np.array(state_matrix), country_names
    
    def _smooth_data(self, data, window=3):
        """
        Apply moving average smoothing to reduce noise
        """
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window)
            end = min(len(data), i + window + 1)
            smoothed.append(np.mean(data[start:end]))
        return smoothed
    
    def get_statistics(self):
        """
        Get basic statistics about the dataset
        
        Returns:
        --------
        dict : Dataset statistics
        """
        if self.raw_data is None:
            return None
        
        stats = {
            'countries': self.raw_data['country'].nunique(),
            'date_range': f"{self.raw_data['date'].min()} to {self.raw_data['date'].max()}",
            'total_records': len(self.raw_data),
            'max_infection_rate': self.raw_data['infected'].max(),
            'avg_infection_rate': self.raw_data['infected'].mean(),
            'max_recovery_rate': self.raw_data['recovered'].max()
        }
        
        return stats