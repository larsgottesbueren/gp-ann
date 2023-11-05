import os
import subprocess

data_path = 'data'

datasets = [
	('turing', 'L2'),
	('deep', 'L2'),
	('text-to-image', 'mips')
]

partitioning_methods = [
	#'GP', 'KMeans',
	'Pyramid',
	'OurPyramid'
]

num_shards_vals = [40, 20, 10]

num_neighbors = 10

build_folders = {
	'L2'   : 'release_l2',
	'mips' : 'release_mips'
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
	print(arglist)
	subprocess.call(arglist)

def compute_all_partitions():
	for dataset, metric in datasets:
		for part_method in partitioning_methods:
			for num_shards in num_shards_vals:
				compute_partition(dataset, metric, part_method, num_shards)

def run_query_set(dataset, metric, part_method, num_shards):
	pfx = os.path.join(data_path, dataset)
	arglist = [build_folders[metric] + '/QueryAttribution',
	           pfx + '_base1B.fbin', pfx + '_query.fbin', pfx + '_ground-truth.bin',
	           str(num_neighbors),
			   pfx + '.partition.k=' + str(num_shards) + '.' + part_method,
			   "output." + dataset + "." + part_method + ".k=" + str(num_shards) + ".csv",
			   part_method,
			   str(num_shards)
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
