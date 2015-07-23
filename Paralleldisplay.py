__author__ = 'bozo'
# Woden Kusner: spherical cap discrepancy 2014-2015 TU Graz


##NEED TO RUN the +- error of <= vs = at the same time inside the loops... right now I run these loops individually

################################################################################
################################################################################
################################################################################

import numpy as np
from joblib import Parallel, delayed
import multiprocessing

################################################################################
################################################################################
################################################################################

maxcores = multiprocessing.cpu_count()

q1 = raw_input("random or from .csv?(r/c):")
put = 1
while put == 1:
    if q1 == "r":
        N = int(raw_input("integer number of random points (default 42):") or 42)
        seed = int(raw_input("integer random seed (default 342343423):") or 342343423)

        def points(N):
            np.random.seed(seed)
            dim = 3
            norm = np.random.normal
            norm_dev = norm(size=(dim, N))
            rad = np.sqrt((norm_dev ** 2).sum(axis=0))
            return np.transpose(norm_dev / rad)

        Z = points(N)
        put = 0

    elif q1 == "c":
        file = str(raw_input("csv file:"))
        import csv

        f = open(file)
        csv_f = csv.reader(f)

        points12 = []
        for row in csv_f:
            points12.append(row)

        Z = np.array(points12).astype('float')
        put = 0

    else:
		q1 = "r"
        #q1 = raw_input("random or .csv? (r/c):")

cores = 1

################################################################################
################################################################################
################################################################################

def NMSQ(u):
    return np.dot(u, u)
# norm squared of u

def NM(u):
    return np.sqrt(np.dot(u, u))
# norm of u

def CRS(u, v):
    return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - v[0] * u[1]]
# cross product of u and v

def VNMal(u, v, w):
    return CRS(u - w, v - w) / NM(CRS(u - w, v - w))
# the center of a three point cap

def LNMal(u, v):
        return (u + v) / NM(u + v)
# the center of a two point cap

def len(u):
    return np.size(u, 0)
# length of a string of points

################################################################################
################################################################################
################################################################################


def DATALOOP(u):
    data = [[0 for i in range(len(u))] for i in range(len(u))]
    vec = [0,0,0]
    area = 0
    hold = 0
    for i in range(0, len(u) - 2):
        for j in range(i + 1, len(u) - 1):
            if NM(u[i] + u[j]) == 0:
                pass
            else:
                y1 = LNMal(u[i], u[j])
                z1 = 0
                # y2 = LNMal(u[i], u[j])
                # z2 = 0

                for l in range(0, len(u)):
                    if np.dot(y1, u[i]) < np.dot(y1, u[l]):
                        z1 = z1 + 1
                    else:
                        z1 = z1

                    # if np.dot(y2, u[i]) <= np.dot(y2, u[l]):
                    #     z2 = z2 + 1
                    # else:
                    #     z2 = z2

                y1d = z1
                y1s = np.floor(len(u)*(1 - np.dot(LNMal(u[i], u[j]), u[i])) / 2).astype(int)
                data[y1d][y1s]=data[y1d][y1s]+1
                # y2d = z2
                # y2s = np.floor(len(u)*(1 - np.dot(LNMal(u[i], u[j]), u[i])) / 2).astype(int)
                # data[y2d][y2s]=data[y2d][y2s]+1

                for k in range(j + 1, len(u)):
                    y3 = VNMal(u[i], u[j], u[k])
                    z3 = 0
                    for l in range(0, len(u)):
                        if np.dot(y3, u[i]) < np.dot(y3, u[l]):
                            z3 = z3 + 1
                        else:
                            z3 = z3
                    y3d = z3
                    y3s = np.floor(len(u)*(1 - np.dot(VNMal(u[i], u[j], u[k]), u[i])) / 2).astype(int)
                    data[y3d][y3s]=data[y3d][y3s]+1
                    # y4 = VNMal(u[i], u[j], u[k])
                    # z4 = 0
                    # for l in range(0, len(u)):
                    #     if np.dot(y4, u[i]) <= np.dot(y4, u[l]):
                    #         z4 = z4 + 1
                    #     else:
                    #         z4 = z4
                    # y4d = z4
                    # y4s = np.floor(len(u)*(1 - np.dot(VNMal(u[i], u[j], u[k]), u[i])) / 2).astype(int)
                    # data[y4d][y4s]=data[y4d][y4s]+1
    return data




from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

#from IPython.Shell import IPShellEmbed
#sh = IPShellEmbed()

data = np.array(DATALOOP(Z))

fig = plt.figure()
ax = Axes3D(fig)

lx= len(data[0])            # Work out matrix dimensions
ly= len(data[:,0])
xpos = np.arange(0,lx,1)    # Set up a mesh of positions
ypos = np.arange(0,ly,1)
xpos, ypos = np.meshgrid(xpos+0.25, ypos+0.25)

xpos = xpos.flatten()   # Convert positions to 1D array
ypos = ypos.flatten()
zpos = np.zeros(lx*ly)

dx = 0.5 * np.ones_like(zpos)
dy = dx.copy()
dz = data.flatten()

ax.bar3d(xpos,ypos,zpos, dx, dy, dz, color='b')

#sh()
#ax.w_xaxis.set_ticklabels(column_names)
#ax.w_yaxis.set_ticklabels(row_names)
#ax.set_xlabel('Dirac?')
#ax.set_ylabel('Sigma?')
#ax.set_zlabel('Count')

plt.show()