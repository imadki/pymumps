import numpy as np
import mumps
from scipy.sparse import csr_matrix
from mpi4py import MPI

comm = MPI.COMM_WORLD

# Set up the test problem:
'''
a = [[ 2.  3.  4.  0.  0.]
 [ 3.  0. -3.  0.  6.]
 [ 0. -1.  1.  2.  0.]
 [ 0.  0.  2.  0.  0.]
 [ 0.  4.  0.  0.  1.]]
'''
n = 5

if comm.Get_rank() == 0:
    irn = np.array([0,1,3,4,1,0,4,2,1,2,0,2], dtype='i')
    jcn = np.array([1,2,2,4,0,0,1,3,4,1,2,2], dtype='i')
    a = np.array([3.0,-3.0,2.0,1.0,3.0,2.0,4.0,2.0,6.0,-1.0,4.0,1.0], dtype='d')
else:
    irn = None
    jcn = None
    a   = None
    
b = np.array([20.0,24.0,9.0,6.0,13.0], dtype='d')

# 12 is the length of irn, jcn and a
counts = int(12/comm.Get_size())

irn_loc = np.zeros(counts, dtype='i')
jcn_loc = np.zeros(counts, dtype='i')
a_loc   = np.zeros(counts, dtype='d')


comm.Scatterv([irn, counts, MPI.INT], irn_loc, root = 0)
comm.Scatterv([jcn, counts, MPI.INT], jcn_loc, root = 0)
comm.Scatterv([a, counts, MPI.DOUBLE], a_loc, root = 0)

ctx = mumps.DMumpsContext(sym=0, par=1)
ctx.set_shape(5)
ctx.set_silent() # Turn off verbose output


ctx.set_distributed_assembled_rows_cols(irn_loc+1, jcn_loc+1)
ctx.set_distributed_assembled_values(a_loc)
ctx.set_icntl(18,3)

if comm.Get_rank() == 0:
    ctx.id.n = 5
    sol = b.copy()
               
#Analyse 
ctx.run(job=1)
#Factorization Phase
ctx.run(job=2)
#Solve

#Allocation size of rhs
if comm.Get_rank() == 0:
    ctx.set_rhs(sol)
else :
    sol = np.zeros(5)

#Solution Phase
ctx.run(job=3)

if ctx.myid == 0:
    print("Solution is %s." % (sol,))

ctx.destroy() # Free memory
