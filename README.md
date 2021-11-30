This code performs the population analysis of binary black holes using the gravitational 
wave data. The methodology is described in the following publications <br />
i) https://arxiv.org/abs/1712.00482 <br />
ii) https://arxiv.org/abs/2006.15047 <br />
iii) https://arxiv.org/abs/2012.08839 <br /><br />
and the scientific results are described in the following publications <br />
i) https://arxiv.org/abs/2011.04502
ii) https://arxiv.org/abs/2111.13991

This code is currently not packaged and need to be checked out <br />
git clone --depth=1 https://github.com/vaibhavtewari/vamana.git

Additional code requirements
============================
Numpy, Scipy, Seaborn, h5py

Folder Organisation
============================
analysis: Define your new analysis here. Import functions that read data, calculate likelihood, and post-process. Also define the range on priors. Initialise hyper-parameters and define the proposal scheme. <br />
gw_data: Data products obtained from the following sources <br />
https://www.gw-openscience.org/O3/O3a/ <br />
https://zenodo.org/record/5636816#.YaK0M_HP3uU <br />
plots: Folder containing saved figures <br />
plotting_nb: Notebook to make plots <br />
results: Files saving posteriors samples and posterior predictives for analysis <br />

Other Files
=============================
script.py: Import data, import analysis in this file. python execution runs the analysis on a dingle CPU. Alternatively, if files are already present in the "temp" folder running script.py will combine those files <br />
other .py files: Source files
local_multicpu.py: Execute multiple independent copies of script.py for faster posterior collection (using CPUs on the local machine)
condor.sub: Submit the analysis on compute nodes using condor (condor_submit condor.sub). 
