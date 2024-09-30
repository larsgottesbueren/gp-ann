import os
import subprocess

data_path = '/global_data/gottesbueren/anns'


metrics = {
    'spacev' : 'L2',
    'sift1B' : 'L2',
    'turing' : 'L2',
    'deep' : 'L2',
    'text-to-image' : 'mips'
}

file_ending = {
    'spacev' : '.i8bin',
    'sift1B' : '.u8bin',
    'deep' : '.fbin',
    'turing' : '.fbin',
    'text-to-image' : '.fbin'
}

datasets = [
    'spacev', 
    'sift1B', 
    'deep',
    'turing', 
    'text-to-image'
]

partitioning_methods = [
    'GP', 
    #'KMeans',
    'BalancedKMeans',
    'OGP',
    #'OGPS',
    'OBKM',
    #'OKM',
    'Pyramid',
    'RKM',
    'ORKM',
    # 'OurPyramid'
]


num_shards_vals = [40]  # , 20, 10]

overlap_values = [0.2]

overlapping_algos = ['OGP', 'OGPS', 'OBKM', 'OKM', 'ORKM']

num_neighbors_values = [1,10,100]

build_folders = {
    'L2': 'release_l2',
    'mips': 'release_mips'
}

def extract_recall(dataset, metric, part_method, num_shards, overlap):
    for num_neighbors in num_neighbors_values:
        pfx = os.path.join(data_path, dataset)
        sfx = ''
        if part_method in overlapping_algos:
            sfx = '.o=' + str(overlap)

        # ground-truth-file routes-file num_neighbors partition-file part-method out-file
        arglist = [build_folders[metric] + '/OracleRecall',
                pfx + '_ground-truth.bin',
                'exp_outputs2/' + dataset + '.' + part_method + '.k=' + str(num_shards) + sfx + '.routes',
                str(num_neighbors),
                pfx + '.partition.k=' + str(num_shards) + '.' + part_method + sfx,
                part_method,
                'exp_outputs2/' + dataset + '.' + part_method + '.k=' + str(num_shards) + sfx + '.nn=' + str(num_neighbors) + '.oracle_recall',
                ]
        print(arglist)
        subprocess.call(arglist)

def run_on_all_datasets(my_func):
    for dataset in datasets:
        metric = metrics[dataset]
        for part_method in partitioning_methods:
            for num_shards in num_shards_vals:
                if part_method not in overlapping_algos:
                    my_func(dataset, metric, part_method, num_shards, 0.0)
                else:
                    for overlap in overlap_values:
                        my_func(dataset, metric, part_method, num_shards, overlap)



def analyze_losses(dataset, metric, part_method, num_shards, overlap):
    if part_method != "GP":
        return
    pfx = os.path.join(data_path, dataset)
    for num_neighbors in num_neighbors_values:
        # points queries ground truth num-neighbors partition part-method out-file
        arglist = [build_folders[metric] + '/AnalyzeApproximationLosses',
                pfx + '_base1B.fbin', pfx + '_query.fbin', pfx + '_ground-truth.bin',
                str(num_neighbors),
                pfx + '.partition.k=' + str(num_shards) + '.' + part_method,
                part_method,
                'exp_outputs2/' + dataset + '.' + part_method + '.k=' + str(num_shards) + '.oracle_recall',
                ]
        print(arglist)
        subprocess.call(arglist)

def convert_spacev_orkm():
    dataset = "spacev"
    metric = metrics[dataset]
    part_method = "ORKM"
    num_shards = 40
    overlap = 0.2

    pfx = os.path.join(data_path, dataset)
    sfx = ''
    if part_method in overlapping_algos:
        sfx = '.o=' + str(overlap)
    

    for num_neighbors in num_neighbors_values:
        ## routes searches ground-truth num-neighbors output-file part-method query-file
        arglist = [build_folders[metric] + '/Convert',
                    'exp_outputs2/' + dataset + '.' + part_method + '.k=' + str(num_shards) + sfx + '.routes',
                    'exp_outputs2/' + dataset + '.' + part_method + '.k=' + str(num_shards) + sfx + '.nn=' + str(num_neighbors) + '.searches',
                    pfx + '_ground-truth.bin',
                    str(num_neighbors),
                    "exp_outputs2/" + dataset + "." + part_method + ".k=" + str(num_shards) + sfx,
                    part_method,
                    pfx + '_query' + file_ending[dataset]
                    ]
        print(arglist)
        subprocess.call(arglist)    

# convert_spacev_orkm()

run_on_all_datasets(extract_recall)
run_on_all_datasets(analyze_losses)
