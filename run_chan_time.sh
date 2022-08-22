#!/usr/bin/env bash

python mctfa/models/indep_chan_time.py data_train.hdf5 groupA 2 indep_chan_time_A.dat
python mctfa/models/indep_chan_time.py data_train.hdf5 groupB 2 indep_chan_time_B.dat
python mctfa/models/indep_chan_time_score.py data_test.hdf5 test 2 indep_chan_time_A.dat indep_chan_time_A_score.dat
python mctfa/models/indep_chan_time_score.py data_test.hdf5 test 2 indep_chan_time_B.dat indep_chan_time_B_score.dat
python mctfa/perf_eval/binary.py truth.txt indep_chan_time_B_score.dat indep_chan_time_A_score.dat indep_chan_time_perf.dat
python mctfa/plotting/binary.py indep_chan_time_perf.dat indep_chan_time
