#!/usr/bin/env bash

python mctfa/models/comb_chan.py data_train.hdf5 groupA comb_chan_A.dat
python mctfa/models/comb_chan.py data_train.hdf5 groupB comb_chan_B.dat
python mctfa/models/comb_chan_score.py data_test.hdf5 test comb_chan_A.dat comb_chan_A_score.dat
python mctfa/models/comb_chan_score.py data_test.hdf5 test comb_chan_B.dat comb_chan_B_score.dat
python mctfa/perf_eval/binary.py truth.txt comb_chan_B_score.dat comb_chan_A_score.dat comb_chan_perf.dat
python mctfa/plotting/binary.py comb_chan_perf.dat comb_chan
