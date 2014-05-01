import numpy as np
import klin

m = 5
data = [[0.011,0.012,0.013,0.014,0.015],
     [0.012,0.022,0.023,0.024,0.025],
     [0.013,0.023,0.033,0.034,0.035],
     [0.014,0.024,0.034,0.044,0.045],
     [0.015,0.025,0.035,0.045,0.055]]

A = np.array(data)

B = np.zeros((m,m))

for i in range(m):
    B[i,i] = i + 1.0

print 'A'
print A
print 'B'
print B

Eval,Evec = klin.pencil(A,B)

print 'Eval is'
print Eval

print 'Evec is'
print Evec



