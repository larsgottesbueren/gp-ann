import os
import subprocess

data_path = 'data'

datasets = [
    ('turing', 'L2'),
    ('deep', 'L2'),
    ('text-to-image', 'mips')
]

partitioning_methods = [
    #'GP', 
    #'KMeans',
    #'BalancedKMeans',
    #'OGP',
    'OGPS',
    #'OBKM',
    #'OKM',
    #'Pyramid',
    # 'OurPyramid'
]

num_shards_vals = [40]  # , 20, 10]

overlap_values = [0.0, 0.2]

overlapping_algos = ['OGP', 'OGPS', 'OBKM', 'OKM']

num_neighbors = 10

build_folders = {
    'L2': 'release_l2',
    'mips': 'release_mips'
}

def create_builds():
    for dist, directory in build_folders.items():
        os.mkdir(directory)
        os.chdir(directory)
        arglist = ['cmake', '..', '-DCMAKE_BUILD_TYPE=Release']
        if dist == 'mips':
            arglist.append('-DMIPS_DISTANCE=ON')
        print(arglist)
        subprocess.call(arglist)
        subprocess.call(['make', '-j'])
        os.chdir('../')
        print('cwd=', os.getcwd())


def download_datasets():
    subprocess.call(['exp_scripts/download_datasets.sh'])


def compute_partition(dataset, metric, part_method, num_shards):
    arglist = [build_folders[metric] + '/Partition',
               os.path.join(data_path, dataset + '_base1B.fbin'),
               os.path.join(data_path, dataset + '.partition'),
               str(num_shards), part_method]
    if part_method in overlapping_algos:
        import copy
        for o in overlap_values:
            my_arglist = copy.copy(arglist)
            my_arglist.append(str(o))
            print(my_arglist)
            subprocess.call(my_arglist)
    else:
        print(arglist)
        subprocess.call(arglist)


def compute_all_partitions():
    for dataset, metric in datasets:
        for part_method in partitioning_methods:
            for num_shards in num_shards_vals:
                compute_partition(dataset, metric, part_method, num_shards)


def run_query_set(dataset, metric, part_method, num_shards, overlap):
    pfx = os.path.join(data_path, dataset)
    sfx = ''
    if part_method in overlapping_algos:
        sfx = '.o=' + str(overlap)
    arglist = [build_folders[metric] + '/QueryAttribution',
               pfx + '_base1B.fbin', pfx + '_query.fbin', pfx + '_ground-truth.bin',
               str(num_neighbors),
               pfx + '.partition.k=' + str(num_shards) + '.' + part_method + sfx,
               "exp_outputs/" + dataset + "." + part_method + ".k=" + str(num_shards) + sfx,
               part_method,
               str(num_shards)
               ]
    print(arglist)
    subprocess.call(arglist)


def run_queries_on_all_datasets():
    for dataset, metric in datasets:
        for part_method in partitioning_methods:
            for num_shards in num_shards_vals:
                if part_method in overlapping_algos:
                    run_query_set(dataset, metric, part_method, num_shards, 0.0)
                else:
                    for overlap in overlap_values:
                        run_query_set(dataset, metric, part_method, num_shards, overlap)


def pareto_filter(dataset, part_method, num_shards):
    pfx = "exp_outputs/" + dataset + "." + part_method + ".k=" + str(num_shards)
    part_file = os.path.join(data_path, dataset) + '.partition.k=' + str(num_shards) + '.' + part_method
    arglist = [build_folders[0] + '/Convert',
               pfx + '.routes', pfx + '.searches', pfx,
               part_method, part_file
               ]
    print(arglist)
    subprocess.call(arglist)


def run_all_pareto_filters():
    for dataset, _ in datasets:
        for part_method in partitioning_methods:
            for num_shards in num_shards_vals:
                pareto_filter(dataset, part_method, num_shards)


compute_all_partitions()
#run_queries_on_all_datasets()
