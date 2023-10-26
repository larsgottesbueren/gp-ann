cd ..
mkdir release_l2 && cd release_l2
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
cd ../ && mkdir release_mips && cd release_mips
cmake .. -DCMAKE_BUILD_TYPE=Release -DMIPS_DISTANCE=ON
make -j
cd ../exp_scripts
