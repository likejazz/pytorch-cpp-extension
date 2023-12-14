#!/bin/bash

mkdir -p bld
cd bld && \
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
  -GNinja .. && \
ninja

cd ..
python test-cmake.py
