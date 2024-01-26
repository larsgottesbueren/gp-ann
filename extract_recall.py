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
    #'OGP',
    #'OGPS',
    #'OBKM',
    #'OKM',
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

def run_on_all_datasets(my_func):
    for dataset, metric in datasets:
        for part_method in partitioning_methods:
            for num_shards in num_shards_vals:
                if part_method not in overlapping_algos:
                    my_func(dataset, metric, part_method, num_shards, 0.0)
                else:
                    for overlap in overlap_values:
                        my_func(dataset, metric, part_method, num_shards, overlap)



def analyze_losses(dataset, metric, part_method, num_shards, overlap):
    pfx = os.path.join(data_path, dataset)
    # points queries ground truth num-neighbors partition part-method out-file
    arglist = [build_folders[metric] + '/AnalyzeApproximationLosses',
               pfx + '_base1B.fbin', pfx + '_query.fbin', pfx + '_ground-truth.bin',
               str(num_neighbors),
               pfx + '.partition.k=' + str(num_shards) + '.' + part_method,
               part_method,
               'exp_outputs/' + dataset + '.' + part_method + '.k=' + str(num_shards) + '.single-center-routes.csv',
               ]
    print(arglist)
    subprocess.call(arglist)

# run_on_all_datasets(extract_recall)
run_on_all_datasets(analyze_losses)
