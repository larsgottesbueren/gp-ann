mkdir data && cd data

#DEEP
wget https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin
mv base.1B.fbin deep_base1B.fbin

wget https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin
mv query.public.10K.fbin deep_query.fbin

wget https://storage.yandexcloud.net/yandex-research/ann-datasets/deep_new_groundtruth.public.10K.bin
mv deep_new_groundtruth.public.10K.bin deep_ground-truth.bin


#Text-to-Image
wget https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin
mv base.1B.fbin text-to-image_base1B.fbin

wget https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin
mv query.public.100K.fbin text-to-image_query.fbin

wget https://storage.yandexcloud.net/yandex-research/ann-datasets/t2i_new_groundtruth.public.100K.bin
mv t2i_new_groundtruth.public.100K.bin text-to-image_ground-truth.bin

# Turing
wget https://comp21storage.blob.core.windows.net/publiccontainer/comp21/MSFT-TURING-ANNS/query100K.fbin
mv query100K.fbin turing_query.fbin

wget https://comp21storage.blob.core.windows.net/publiccontainer/comp21/MSFT-TURING-ANNS/query_gt100.bin
mv query_gt100.bin turing_ground-truth.bin

echo Downloading the base vectors from Microsoft via wget/curl is excruciatingly slow. 
echo Check out the website https://big-ann-benchmarks.com/neurips21.html for a faster way to download this dataset -- using AzCopy.

wget https://comp21storage.blob.core.windows.net/publiccontainer/comp21/MSFT-TURING-ANNS/base1b.fbin
mv base1b.fbin turing_base1B.fbin

cd ..
