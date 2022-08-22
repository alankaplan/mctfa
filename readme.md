# Multi-Channel Time-Frequency Analysis (MCTFA)

Software for training and evaluating models on multi-channel time series data, such as electrocorticography.


## Models

This software includes training and evaluation code for three types of models:

1. Single time-frequeny model (models/comb_chan). This model assumes all channels have the same spectrum.
2. Independent channel time-frequency model (models/indep_chan). This model assumes each channel has a seperate spectral model.
3. Independent channel + time model (models/indep_chan_time). This model has seperate spectral models for each channel that are also time dependent.

## Performance Evaluation

Scoring can be done using the score files (models/comb_chan_score, models/indep_chan_score, models/indep_chan_time_score) for each model type. Performance evaluation is performed using perf_eval/binar.py and ROC plots can be gnereated using plotting/binary.py.

## Examples

Generate synthetic data using make_data.py. This will generate data for two classes with 20 events, 100 channels, and 20 seconds of 500 Hz data for each channel. All of the channels contain white noise except for channel #49, which has different spectra for the two classes.

    export PYTHONPATH='/path/to/mctfa'
    python make_data.py

Train single spectum models by running ./run_comb.sh. This will generate one model for each class and perform performance evaluation.

Train independent channel models by running ./run_chan.sh

## Reference

See the following paper for more details on the model:

Kaplan, A. D., Q. Cheng, P. Karande, E. Tran, M. Bijanzadeh, H. Dawes, and E. Chang. 2019. “Localization of Emotional Affect in Electrocorticography Using a Model Based Discrimination Measure.” In 2019 53rd Asilomar Conference on Signals, Systems, and Computers, 1709–13.

## Release

LLNL-CODE-838872
MCTFA is distributed under the terms of the MIT license. All new contributions must be made under this license.
SPDX-License-Identifier: MIT
