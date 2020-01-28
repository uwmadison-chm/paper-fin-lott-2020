# Thukdam analysis scripts

This is the 2019 conversion to python and the MNE library by Dan Fitch.

In progress.

## Installation

    conda env create -f environment.yaml
    conda activate thukdam
    pip install mne

On windows, you may need to:

    pip install pyqt5

## Usage: MMN

This MNE-based analysis does NOT require decimation or cropping of input 
files, and will attempt to locate the MMN events automatically.

If you're logged into the BI servers, you can just run:

    set_study thukdam
    mmn.py </path/to/FILENAME.bdf> --view

If running locally, you can put it in your path or run like

    conda activate thukdam
    python3 /this/directory/mmn.py </path/to/FILENAME.bdf> --view

### MMN Options

To view all the options, run:

    mmn.py --help

### Artifact Rejection

If you run on a BDF file that's properly filed in `/study/thukdam/raw-data/subjects`,
the intermediate artifact mask and event timing metadata will be loaded from a 
matching path in `/study/thukdam/intermediate_data/eeg/subjects`.

So, you can run multiple times to progressively reject artifacts, review, and 
produce graphs.
