echo "Preparing Python execution"
source /export/home/ldittmer/.bashrc
source /export/home/ldittmer/miniconda3/etc/profile.d/conda.sh
conda activate pyscf
source /opt/software/intel/compilers_and_libraries_2020/bin/compilervars.sh -arch intel64 -platform linux
source /opt/software/intel/compilers_and_libraries_2020/compilers_and_libraries_2020.4.304/linux/mkl/bin/mklvars.sh intel64
export PYTHONPATH="$PYTHONPATH:/export/home/ldittmer/Documents/Linus/FDIIS/pyscf_workspace/pyscf"
export LD_PRELOAD="$MKLROOT/lib/intel64/libmkl_core.so:$MKLROOT/lib/intel64/libmkl_sequential.so"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYSCF_MAX_MEMORY=1000000
#export PYSCF_TMP_DIR=${SCRATCH_DIR}

