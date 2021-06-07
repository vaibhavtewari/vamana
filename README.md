This code is currently not packaged and need to be checked out <br />
git clone https://github.com/vaibhavtewari/vamana.git

Additional code requirements
============================
Numpy, Scipy, Seaborn, Pathos

Folder Organisation
============================
analysis: Define your new analysis here. Code contains two examples <br />
gw_data: Data obtained from the following sources <br />
https://dcc.ligo.org/LIGO-P2000434/public <br />
https://www.gw-openscience.org/O3/O3a/ <br />
plots: Folder containing saved figures <br />
plotting_nb: Notebook to make plots <br />
reference_distribution_nb: Folder containing notebooks to obtain the reference chirp mass distribution <br />
results: Posteriors samples for analysis <br />
submit: Scripts to run independent copies of code for faster posterior collection 

Other Files
=============================
debug.ipynb: Notebook to test an analysis <br />
.py: Source files