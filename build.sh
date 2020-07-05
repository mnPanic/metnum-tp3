#!/bin/sh

# cleanup
mkdir build
cd build
rm -rf *

# corro el CMakeLists del root
cmake \
  -DPYTHON_EXECUTABLE="$(which python)" \
  -DCMAKE_BUILD_TYPE=Release ..

# instalo
make install