__author__ = 'bozo'
# Woden Kusner: spherical cap discrepancy 2014-2015 TU Graz


##NEED TO RUN the +- error of <= vs = at the same time inside the loops... right now I run these loops individually

################################################################################
################################################################################
################################################################################

import numpy as np
from joblib import Parallel, delayed

################################################################################
################################################################################
################################################################################


q1 = raw_input("random or .csv? r/c:")
put = 1
while put == 1:
    if q1 == "r":
        N = int(raw_input("integer number of random points:") or 42)
        seed = int(raw_input("integer random seed:") or 342343423)

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
        q1 = raw_input("random or .csv? r/c:")

#cores = raw_input("cores to run (choose 1 for non-parallel, 2-4 for parallel):")

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


def FULLDISCREPANCYLOOP(u):
    hold = 0
    for i in range(0, len(u) - 2):
        for j in range(i + 1, len(u) - 1):
            if NM(u[i] + u[j]) == 0:
                hold=hold
            else:
                y1 = LNMal(u[i], u[j])
                z1 = 0
                y2 = LNMal(u[i], u[j])
                z2 = 0

                for l in range(0, len(u)):
                    if np.dot(y1, u[i]) < np.dot(y1, u[l]):
                        z1 = z1 + 1
                    else:
                        z1 = z1

                    if np.dot(y2, u[i]) <= np.dot(y2, u[l]):
                        z2 = z2 + 1
                    else:
                        z2 = z2

                y1d = z1 / (len(u) - 0.0)
                y1s = (1 - np.dot(LNMal(u[i], u[j]), u[i])) / 2
                if np.absolute(y1d - y1s) > hold:
                    hold = np.absolute(y1d - y1s)
                else:
                    hold = hold

                y2d = z2 / (len(u) - 0.0)
                y2s = (1 - np.dot(LNMal(u[i], u[j]), u[i])) / 2
                if np.absolute(y2d - y2s) > hold:
                    hold = np.absolute(y2d - y2s)
                else:
                    hold = hold

                for k in range(j + 1, len(u)):
                    y3 = VNMal(u[i], u[j], u[k])
                    z3 = 0
                    for l in range(0, len(u)):
                        if np.dot(y3, u[i]) < np.dot(y3, u[l]):
                            z3 = z3 + 1
                        else:
                            z3 = z3
                    y3d = z3 / (len(u) + 0.0)
                    y3s = (1 - np.dot(VNMal(u[i], u[j], u[k]), u[i])) / 2
                    if np.absolute(y3d - y3s) > hold:
                        hold = np.absolute(y3d - y3s)
                    else:
                     hold = hold

                    y4 = VNMal(u[i], u[j], u[k])
                    z4 = 0
                    for l in range(0, len(u)):
                        if np.dot(y4, u[i]) <= np.dot(y4, u[l]):
                            z4 = z4 + 1
                        else:
                            z4 = z4
                    y4d = z4 / (len(u) + 0.0)
                    y4s = (1 - np.dot(VNMal(u[i], u[j], u[k]), u[i])) / 2
                    if np.absolute(y4d - y4s) > hold:
                        hold = np.absolute(y4d - y4s)
                    else:
                        hold = hold
    return hold




################################################################################
################################################################################
################################################################################

def FULLDISCREPANCYITERATOR(i):
    hold = 0
    u = Z
    for j in range(i + 1, len(u) - 1):

        y1 = LNMal(u[i], u[j])
        z1 = 0
        y2 = LNMal(u[i], u[j])
        z2 = 0

        for l in range(0, len(u)):
            if np.dot(y1, u[i]) < np.dot(y1, u[l]):
                z1 = z1 + 1
            else:
                z1 = z1

            if np.dot(y2, u[i]) <= np.dot(y2, u[l]):
                z2 = z2 + 1
            else:
                z2 = z2

        y1d = z1 / (len(u) - 0.0)
        if NM(u[i] + u[j]) == 0:
            y1s = 0.5
        else:
            y1s = (1 - np.dot(LNMal(u[i], u[j]), u[i])) / 2
        if np.absolute(y1d - y1s) > hold:
            hold = np.absolute(y1d - y1s)
        else:
            hold = hold

        y2d = z2 / (len(u) - 0.0)
        if NM(u[i] + u[j]) == 0:
            y2s = 0.5
        else:
            y2s = (1 - np.dot(LNMal(u[i], u[j]), u[i])) / 2
        if np.absolute(y2d - y2s) > hold:
            hold = np.absolute(y2d - y2s)
        else:
            hold = hold

        for k in range(j + 1, len(u)):
            y3 = VNMal(u[i], u[j], u[k])
            z3 = 0
            for l in range(0, len(u)):
                if np.dot(y3, u[i]) < np.dot(y3, u[l]):
                    z3 = z3 + 1
                else:
                    z3 = z3
            y3d = z3 / (len(u) + 0.0)
            y3s = (1 - np.dot(VNMal(u[i], u[j], u[k]), u[i])) / 2
            if np.absolute(y3d - y3s) > hold:
                hold = np.absolute(y3d - y3s)
            else:
                hold = hold

            y4 = VNMal(u[i], u[j], u[k])
            z4 = 0
            for l in range(0, len(u)):
                if np.dot(y4, u[i]) <= np.dot(y4, u[l]):
                    z4 = z4 + 1
                else:
                    z4 = z4
            y4d = z4 / (len(u) + 0.0)
            y4s = (1 - np.dot(VNMal(u[i], u[j], u[k]), u[i])) / 2
            if np.absolute(y4d - y4s) > hold:
                hold = np.absolute(y4d - y4s)
            else:
                hold = hold
    return hold


def PARALLELDISCREPANCY(u):
    return Parallel(n_jobs=2)(delayed(FULLDISCREPANCYITERATOR)(i) for i in range(len(u) - 2))

################################################################################
################################################################################
################################################################################    

#print np.max(PARALLELDISCREPANCY(Z))

#print Z



def FULLDISCREPANCYIT(i):
    hold = 0
    u = Z
    for j in range(i + 1, len(u) - 1):
        if NM(u[i] + u[j]) == 0:
            hold=hold
        else:
            y1 = LNMal(u[i], u[j])
            z1 = 0
            y2 = LNMal(u[i], u[j])
            z2 = 0

            for l in range(0, len(u)):
                if np.dot(y1, u[i]) < np.dot(y1, u[l]):
                    z1 = z1 + 1
                else:
                    z1 = z1

                if np.dot(y2, u[i]) <= np.dot(y2, u[l]):
                    z2 = z2 + 1
                else:
                    z2 = z2

            y1d = z1 / (len(u) - 0.0)
            y1s = (1 - np.dot(LNMal(u[i], u[j]), u[i])) / 2
            if np.absolute(y1d - y1s) > hold:
                hold = np.absolute(y1d - y1s)
            else:
                hold = hold

            y2d = z2 / (len(u) - 0.0)
            y2s = (1 - np.dot(LNMal(u[i], u[j]), u[i])) / 2
            if np.absolute(y2d - y2s) > hold:
                hold = np.absolute(y2d - y2s)
            else:
                hold = hold

            for k in range(j + 1, len(u)):
                y3 = VNMal(u[i], u[j], u[k])
                z3 = 0
                for l in range(0, len(u)):
                    if np.dot(y3, u[i]) < np.dot(y3, u[l]):
                        z3 = z3 + 1
                    else:
                        z3 = z3
                y3d = z3 / (len(u) + 0.0)
                y3s = (1 - np.dot(VNMal(u[i], u[j], u[k]), u[i])) / 2
                if np.absolute(y3d - y3s) > hold:
                    hold = np.absolute(y3d - y3s)
                else:
                    hold = hold

                y4 = VNMal(u[i], u[j], u[k])
                z4 = 0
                for l in range(0, len(u)):
                    if np.dot(y4, u[i]) <= np.dot(y4, u[l]):
                        z4 = z4 + 1
                    else:
                        z4 = z4
                y4d = z4 / (len(u) + 0.0)
                y4s = (1 - np.dot(VNMal(u[i], u[j], u[k]), u[i])) / 2
                if np.absolute(y4d - y4s) > hold:
                    hold = np.absolute(y4d - y4s)
                else:
                    hold = hold
    return hold

def PARALLEL(u):
    return np.max(Parallel(n_jobs=2)(delayed(FULLDISCREPANCYIT)(i) for i in range(len(u) - 2)))




print PARALLEL(Z)