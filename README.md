# Disease Spread Modeling using Linear Algebra

## Overview
This project models the spread of a disease using the SIR (Susceptible–Infected–Recovered) model combined with linear algebra concepts such as matrices, eigenvalues, and Singular Value Decomposition (SVD).

Using real-world COVID-19 data, the project demonstrates how Markov chains and matrix operations can be used to analyze system dynamics, steady-state behavior, and dimensionality reduction.

---

## Objectives
- Model disease spread using the SIR framework  
- Represent transitions between states using a Markov transition matrix  
- Analyze system stability using eigenvalues and eigenvectors  
- Apply SVD to understand data structure and reduce dimensionality  
- Visualize infection trends, transition probabilities, and predictions  

---

## Key Concepts
- SIR Model (Susceptible, Infected, Recovered)  
- Markov Chains  
- Transition Matrix  
- Eigenvalues and Eigenvectors  
- Steady-State Analysis  
- Singular Value Decomposition (SVD)  
- Data Normalization and Visualization  

---

## Features
- Infection trends over time for multiple countries  
- Current infection comparison across countries  
- State transition matrix visualization  
- Eigenvalue spectrum for system dynamics  
- Singular value spectrum analysis  
- Cumulative explained variance for dimensionality reduction  
- Prediction of future infection trends  

---

## Project Structure
<img width="311" height="400" alt="image" src="https://github.com/user-attachments/assets/9f713fc9-7b34-46ff-a941-8e2830817af1" />

---

## Install dependencies:
pip install -r requirements.txt

---

## Usage
Run the project:
python main.py

---

This will:
- Process COVID-19 data  
- Construct the transition matrix  
- Perform SVD and eigenvalue analysis  
- Generate all visualizations  

---

## Results and Insights
- The dataset shows a low-rank structure, meaning most variation can be captured using a small number of components  
- Eigenvalue analysis indicates fast system convergence  
- The transition matrix highlights dominant state behavior  
- Steady-state results reflect long-term system dynamics  

---

## Limitations
- Transition matrix estimation may introduce bias due to simplified assumptions  
- The Markov model assumes memoryless transitions, which may not fully capture real-world dynamics  
- Prediction results depend on data quality and completeness  

---

## Future Improvements
- Improve transition estimation using full probabilistic modeling  
- Incorporate additional real-world factors such as vaccination and policy changes  
- Enhance prediction accuracy using advanced time-series methods  
- Add interactive visualizations  

---

## Technologies Used
- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  

---

## Author
Sakshi Ghodke

---
