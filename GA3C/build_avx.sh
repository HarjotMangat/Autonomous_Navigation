#!/bin/bash

CPU_CORES=$(grep -c ^processor /proc/cpuinfo)

mkdir ga3c_Csim/pysim2d
cd ga3c_Csim/pysim2d
cmake ../../../Simulation2d -DCMAKE_BUILD_TYPE=Release -DWITH_AVX=ON && make -j $CPU_CORES