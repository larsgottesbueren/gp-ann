cd ..
mkdir release && cd release
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
cd ../exp_scripts

./download_datasets.sh
./compute_partitions.sh
./bench.sh
./build_all_plots.sh
