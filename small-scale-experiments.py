import os
import subprocess

data_path = 'data/'

datasets = [
    ('sift', 'L2'),
    ('glove', 'mips') # actually angular, but this is equivalent if normalized
]

partitioning_methods = [
    'GP',
    'RKM',
    #'KMeans',
    'BalancedKMeans',
]

num_shards_vals = [16]

num_neighbors = 10

build_folders = {
    'L2': 'release_l2',
    'mips': 'release_mips'
}

def compute_partition(dataset, metric, part_method, num_shards):
    arglist = [build_folders[metric] + '/Partition',
               os.path.join(data_path, dataset + '.fbin'),
               os.path.join(data_path, dataset + '.partition'),
               str(num_shards), part_method, 'strong']
    print(arglist)
    subprocess.call(arglist)

def compute_all_partitions():
    for dataset, metric in datasets:
        for part_method in partitioning_methods:
            for num_shards in num_shards_vals:
                compute_partition(dataset, metric, part_method, num_shards)


def run_query_set(dataset, metric, part_method, num_shards):
    pfx = os.path.join(data_path, dataset)
    sfx = ''
    arglist = [build_folders[metric] + '/SmallScaleQueries',
               pfx + '.fbin', pfx + '.query.fbin', pfx + '.ground_truth.bin',
               str(num_neighbors),
               pfx + '.partition.k=' + str(num_shards) + '.' + part_method + sfx,
               part_method,
               "exp_outputs/" + dataset + "." + part_method + ".k=" + str(num_shards) + '.csv'
               ]
    print(arglist)
    subprocess.call(arglist)


def run_queries_on_all_datasets():
    for dataset, metric in datasets:
        for part_method in partitioning_methods:
            for num_shards in num_shards_vals:
                run_query_set(dataset, metric, part_method, num_shards)


compute_all_partitions()
run_queries_on_all_datasets()
