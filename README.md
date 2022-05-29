# MatrixBiasReduction
Bias reduction in covariance matrices for high dimensionality portfolios.
This work is from my thesis <Only available in Spanish>. Reducción de sesgo en matrices de covarianza y sus efectos en portafolios de alta dimensionalidad.
Also thanks to the work from: 
  - Ledoit and Wolf, for their "Numerical implementation of the QuEST function" available at: https://www.econ.uzh.ch/en/people/faculty/wolf/publications.html#Journal_Papers
  - The folks at sci-kit learn for the Linear Shrinkage implementation, included with the python package. For more info check: https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html
  - To my thesis director Andres Garcia Medina, for the general code direction.

Note: This code works with Python 3.8 and on Windows or Linux machines of the gate.
  
The main function in the code does the following:
  - Reads previously collected financial returns data or generates it. 
  - Splits the data into in sample and out of sample data.
  - It calculates and saves bias reduced covariance matrices (in-sample and out-sample).
  - It calculates metrics to determine the bias reduction with the in-sample data vs the true covariance (with the real data the out-sample is used to create a simulation for the true covariance).
  - It also calculates financial metrics for the portfolio performance.

  Then a secondary function (GraphFF) will create efficient frontiers for the matrices, for a proper visualization of the results.
  This code also includes other codes mainly for data collection (WebScraper) and data cleaning (Preprocess) for the real scenarios. 
