import numpy as np
import klin


data = [[0.011,0.012,0.013,0.014,0.015],
     [0.012,0.022,0.023,0.024,0.025],
     [0.013,0.023,0.033,0.034,0.035],
     [0.014,0.024,0.034,0.044,0.045],
     [0.015,0.025,0.035,0.045,0.055]]

A = np.array(data)

Eval,Evec = klin.eigen(A)

print 'Eval is'
print Eval

print 'Evec is'
print Evec



