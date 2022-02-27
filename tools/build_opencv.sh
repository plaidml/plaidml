#!/bin/bash

BUILD_DIR=$1
INSTALL_DIR=$2

rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.4.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.4.0.zip
#wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
#wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
unzip opencv.zip
unzip opencv_contrib.zip
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} -DWITH_PNG=OFF -DWITH_JPEG=OFF -DWITH_TIFF=OFF -DWITH_GSTREAMER=OFF -DBUILD_opencv_java=OFF -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.4.0/modules ../opencv-4.4.0
#cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules ../opencv-4.x
cmake --build . -j 16
make install
cd ${INSTALL_DIR}
if [ -d "./lib64" ]
then
    echo "Directory ./lib64 exists."
else
    echo "Directory ./lib64 does not exist"
    mkdir lib64
    cd lib64
    echo "./lib64 has been created"
    cp -P ./../lib/libopencv_* .
    echo "libopencv* files have been copied to ./lib64"
    cp -R ./../lib/cmake .
    echo "cmake files have been copied to ./lib64"
fi
