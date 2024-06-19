# Running mdestimates within a conda environment

## Set up a conda environment 

1. Make sure you have Anaconda installed on your computer.
```
conda list
```
2. Make sure the default channel is up-to-date
```
conda update -n base -c defaults conda
```
3. Add the conda-forge channel (ignore the warning if it is already present)
```
conda config --add channels conda-forge
```
4. Create a new conda environment
```
conda create --name mdenv python=3.9
```
5. Activate your environment
```
conda activate mdenv
```
6. Install a few important libraries
```
conda install git jupyter notebook
```

## Install quadpy
Unfortunately, `quadpy` is now proprietary. However, we can install the latest open source version.
```
conda install conda-forge::quadpy==0.16.10
```

## Install PorePy
1. Head to a folder where you wish to clone PorePy, clone it, and checkout commit `67b32246`:
```
git clone https://github.com/pmgbergen/porepy.git
cd porepy
git fetch --all
git checkout 67b32246
```
2. Install PorePy requirements using conda
```
conda install --file requirements_dev.txt
```
3. Install PorePy using pip
```
pip install --user -e .
```
4. Check your PorePy installation
```
pytest
```

## Install mdestimates
Head to the same folder you've installed PorePy, clone the development version of `mixdim-estimates` and install using pip:
```
git clone https://github.com/jhabriel/mixdim-estimates.git
cd mixdim-estimates
pip install --user -e .
```
