import os
import subprocess

data_path = '/global_data/gottesbueren/anns'

datasets = [
    ('turing', 'L2'),
    ('deep', 'L2'),
    ('text-to-image', 'mips')
]

partitioning_methods = [
    'GP',
    'KMeans',
    'BalancedKMeans',
    'OGP',
    'OGPS',
    'OBKM',
    'OKM',
    'Pyramid',
    # 'OurPyramid'
]

num_shards_vals = [40]  # , 20, 10]

overlap_values = [0.2]

overlapping_algos = ['OGP', 'OGPS', 'OBKM', 'OKM']

num_neighbors = 10

build_folders = {
    'L2': 'release_l2',
    'mips': 'release_mips'
}

def extract_recall(dataset, metric, part_method, num_shards, overlap):
    pfx = os.path.join(data_path, dataset)
    sfx = ''
    if part_method in overlapping_algos:
        sfx = '.o=' + str(overlap)

    # ground-truth-file routes-file num_neighbors partition-file part-method out-file
    arglist = [build_folders[metric] + '/OracleRecall',
               pfx + '_ground-truth.bin',
               'exp_outputs/' + dataset + '.' + part_method + '.k=' + str(num_shards) + sfx + '.routes',
               str(num_neighbors),
               pfx + '.partition.k=' + str(num_shards) + '.' + part_method + sfx,
               part_method,
               'exp_outputs/' + dataset + '.' + part_method + '.k=' + str(num_shards) + sfx + '.oracle_recall',
               ]
    print(arglist)
    subprocess.call(arglist)


def run_extract_on_all_datasets():
    for dataset, metric in datasets:
        for part_method in partitioning_methods:
            for num_shards in num_shards_vals:
                if part_method not in overlapping_algos:
                    extract_recall(dataset, metric, part_method, num_shards, 0.0)
                else:
                    for overlap in overlap_values:
                        extract_recall(dataset, metric, part_method, num_shards, overlap)



run_extract_on_all_datasets()
