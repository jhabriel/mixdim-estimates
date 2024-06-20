# Running mixdim-estimates using conda

This is set of instructions on how to set up a `conda` environment to install an run the `mixdim-estimates` package.

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
conda create --name mdenv python=3.10
```
5. Activate your environment
```
conda activate mdenv
```

## Install critical packages using conda

Some libraries **must** be installed using `conda`. The rest can be installed using `pip`
```
conda install numpy=1.21.2 scipy=1.7.3 quadpy=0.16.10 'matplotlib<3.6' jupyter notebook
```

## Install PorePy

1. Go to `https://github.com/pmgbergen/porepy.git` and manually download (as a zip) the entire PorePy repository corresponding to commit `67b32246`. This is a somewhat old version of PorePy, we are now currently working on updating `mdestimates` to be in sync with the latest PorePy version.
2. Rename the folder to `porepy_mde` and move it inside a folder named `summer_school_mixdim`.
3. In your terminal, go inside the `porepy_mde` folder.
4. Install the remaining required libraries using `pip`:
   ```
   pip install -r -requirements_dev.txt
   ```
5. Now, install PorePy in user-editable mode:
   ```
   pip install --user -e .
   ```

## Install `mdestimates`

1. Head to the `summer_school_mixdim` folder once again.
2. Clone the `mdestimates` repository:
   ```
   git clone https://github.com/jhabriel/mixdim-estimates.git
   ```
3. Navigate inside the `mixdim-estimates` folder
   ```
   cd mixdim-estimates
   ```
4. Install `mdestimates` in user-editable mode:
   ```
   pip install --user -e .
   ```

## Run the tutorial

1. In your terminal, type
   ```
   jupyter notebook
   ```
   This should open your default browser.
2. Go to `tutorials` and open `basic_tutorial.ipynb`.
3. You should be able to run the notebook withouth any errors.
