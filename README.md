This code performs the population analysis of binary black holes using the gravitational 
wave data. The methodology is described in the following publications <br />
i) https://iopscience.iop.org/article/10.1088/1361-6382/aac89d <br />
ii) https://iopscience.iop.org/article/10.1088/1361-6382/ac0b54 <br />
and the scientific results are described in the following publications <br />
i) https://iopscience.iop.org/article/10.3847/2041-8213/abfbe7 <br />
ii) https://iopscience.iop.org/article/10.3847/1538-4357/ac589a <br />
iii) https://academic.oup.com/mnras/article/527/1/298/7317695 <br />

This code is currently not packaged and need to be checked out <br />
git clone https://github.com/vaibhavtewari/vamana.git

Additional code requirements
============================
Numpy, Scipy, Seaborn, h5py

Folder Organisation
============================
analysis: Define your new analysis here. Code contains two examples <br />
gw_data: Data obtained from the following sources <br />
https://dcc.ligo.org/LIGO-P2000434/public <br />
https://www.gw-openscience.org/O3/O3a/ <br />
plots: Folder containing saved figures <br />
plotting_nb: Notebook to make plots <br />
results: Posteriors samples for analysis <br />
submit: Scripts to run independent copies of code for faster posterior collection 

Other Files
=============================
debug.ipynb: Notebook to test an analysis <br />
.py: Source files

To Submit
=============================
debug.ipynb can run a single thread run <br />
For multiple threads do: python submit/relevant analysis(2 submit files are provided) <br />
If files are already present in the "temp" folder instead of running the analysis code will combine these!
