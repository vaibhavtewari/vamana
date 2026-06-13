from vamana.utils import combine_chains, load_samples

# Combine all chains into a single HDF5 file
output='posteriors/GWTC3.hdf5'
combine_chains(
    pattern='./temp/*.h5',
    output=output
)