# -*- coding: utf-8 -*-
"""
Created on Fri May  2 08:27:25 2014

@author: kurita
"""

import numpy as np
import MySU3

data = np.loadtxt('tsu3.dat', comments='#')

E, U, V = MySU3.su3(data)

print 'data\n', data

print 'E\n', E

print 'U\n', U

print 'V\n', V

print '====== transposed data ======='

E, U, V = MySU3.su3(data.T)

print 'data\n', data

print 'E\n', E

print 'U\n', U

print 'V\n', V
