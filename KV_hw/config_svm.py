#!/usr/bin/python
#Filename: myconfig.py

import numpy as np
import contextlib


Eps = 10**(-10)

np.set_printoptions(precision=3, linewidth=1000, suppress=True, threshold=20)

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield 
    np.set_printoptions(**original)


#exec while import\\
import matplotlib as mpl
from matplotlib import rc

rc('figure', figsize=(12, 6)) 
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True) 
#rc('text.latex', unicode=True) 
#rc('text.latex', preamble=r'\usepackage[utf8]{inputenc}')
#rc('text.latex', preamble=r'\usepackage[russian]{babel}')
#rc('text.latex', preamble=r'\usepackage{amsmath}')
rc('axes', labelsize=16, titlesize=20)
rcParams = mpl.rcParams
#font = {'family': 'sans-serif',
#'weight': 'normal', 'size': 22}
#rc('font', **font)

pic_dir = 'pic/'
