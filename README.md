# Analysis code for Lott, et. al 2020, Frontiers in Neuroscience 

Paper title: "Apparent Attenuation of Postmortem Decomposition among Tibetan Buddhist Meditators Not Accompanied by Detectable Electroencephalographic Activity 24-hours After Clinical Declaration of Death"

[![DOI](https://zenodo.org/badge/285621487.svg)](https://zenodo.org/badge/latestdoi/285621487)

Analysis code for Biosemi EEG processing, doing artifact rejection,
and various statistical summaries for the above paper.

This is the 2019-20 conversion of prior analysis code written in Matlab and 
EEGLAB to Python and the [MNE](https://mne.tools/) library,
coded by Dan Fitch with support from John Koger.

This was developed in somewhat of a rush and also not assuming we would be
basing future EEG studies on this code, so there's quite a bit of duplication
and copy-paste instead of generalizing. Apologies to future folk referencing 
this!


## Data structure and context

Subject data is pulled from `/study/thukdam/raw-data/subjects`, and
EEG data should be in `biosemi`.

Intermediate and final results go to `/study/thukdam/analyses`.

NOTE: That subjects dir has "index" links generated by scripts a level up from here,
see `../rebuild_subject_links.py` which reads from `/study/thukdam/subject_metadata.xlsx`

## Installation

    conda env create -f environment.yaml
    conda activate thukdam
    pip install mne

On windows, you may need to:

    pip install pyqt5

## Usage: MMN artifact rejection and viewing

This MNE-based analysis does NOT require decimation or cropping of input 
files, and will attempt to locate the MMN events automatically.

If you're logged into the BI servers, you can just run:

    set_study thukdam
    mmn.py </path/to/FILENAME.bdf>

If running locally, you can put it in your path or run like

    conda activate thukdam
    python3 /this/directory/mmn.py </path/to/FILENAME.bdf> --view

### MMN Options

To view all the options, run:

    mmn.py --help

### Artifact Rejection

If you run on a BDF file that's properly filed in `/study/thukdam/raw-data/subjects`,
the intermediate artifact mask and event timing metadata will be loaded from a 
matching path in `/study/thukdam/analyses/eeg_artifacts`.

So, you can run multiple times to progressively reject artifacts, review, and 
produce graphs.

## ABR

Very similar to MMN above.

    set_study thukdam
    abr.py </path/to/FILENAME.bdf> 

### ABR Options

To view all the options, run:

    abr.py --help


# Analysis

## MMN

### Grand average

Grand average analysis is with `mmn_grand_average.py`. You pass it a list of
the participant IDs, like `mmn_grand_average.py FM1618 FM1721 ...`

By default, writes cached averages to `/study/thukdam/analyses/eeg_statistics/`
and plots to `/study/thukdam/analyses/eeg_plots/` in a subdirectory with the
date and time, or you can name your output directory with `--name FOLDERNAME`.

### Per-participant and group comparison statistics

See `mmn_analysis.py`


## ABR

### Grand average

Grand average analysis is with `abr_grand_average.py`, and works pretty much
identically to the MMN grand average above.


### Per-participant and group comparison statistics

See `abr_analysis.py`

