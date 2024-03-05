import os, subprocess

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

create_builds()