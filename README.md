This code is currently not packaged and need to be checked out
git clone https://github.com/vaibhavtewari/vamana.git

Additional code requirements
============================
Numpy, Scipy, Seaborn, Pathos

Folder Organisation
============================
analysis: Define your new analysis here. Code contains two examples
gw_data: Data obtained from the following sources
https://dcc.ligo.org/LIGO-P2000434/public
https://www.gw-openscience.org/O3/O3a/
plots: Folder containing saved figures
plotting_nb: Notebook to make plots
reference_distribution_nb: Folder containing notebooks the reference chirp mass distribution
results: Posteriors samples for analysis
submit: Scripts to run independent copies of code for faster posterior collection

Other Files
=============================
debug.ipynb: Notebook to test an analysis
.py: Source files