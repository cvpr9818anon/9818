import sys
import os

path_root = sys.argv[1]
path_suffixes = sys.argv[2:]

for path_suffix in path_suffixes:
    lst = []
    for i in xrange(0,100):
        fpath = path_root + '_' + path_suffix + '_' + str(i) + '/cls.txt'
        if not os.path.exists(fpath): continue
        with open(fpath,'r') as f:
            lines = f.readlines()
            elem = lines[-1]
            js = float(elem)
        fpath = path_root + '_' + path_suffix + '_' + str(i) + '/fid.txt'
        if not os.path.exists(fpath): continue
        with open(fpath,'r') as f:
            fid = float(f.readline())
        lst.append((fid,js))
    print path_suffix.replace('.', '') + ' = ', lst

