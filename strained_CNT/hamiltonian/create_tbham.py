import numpy as np

# from fonksiyonlar import *

import matplotlib.pyplot as plt

from scipy.sparse import bmat


# =============================================================================

#  Here we add arbitrary value as lead source term

# =============================================================================


R = 36  # num of atoms around zigzag, for convention this can only be even number

# R is not total num of atoms in one cell, 2R is.

Q = 1  # this is num of cell,  each cell has 2R atoms


# Then Psi is [r1, r2, r3 ... rR, rR+1, rR+2, ... r2R]

# sideways bonds will be on even numbered r's

epsilon = 0  # eV   #onsite energy of carbon

t = -2.7  # eV hopping term between carbons


H = np.zeros((2 * R, 2 * R))

np.fill_diagonal(H, epsilon)


# first atom is indexed 0

# add hoppings

for i in range(R - 1):  # from zero to R-2         # CHECKED
    H[i, i + 1] = t

    H[i + 1, i] = t

    # dont connect index R-1 and R

    H[i + R, i + R + 1] = t

    H[i + R + 1, i + R] = t


# sideways connections                          # CHECKED

for i in range(R):  # from zero to R-1
    if i % 2 == 0:  # if even
        H[i, i + R] = t

        H[i + R, i] = t


# connect bottom of layer to top, to ensure connectivity

H[0, R - 1] = t

H[R - 1, 0] = t

H[R, 2 * R - 1] = t

H[2 * R - 1, R] = t


# CELL IS DONE   H is one cell


# T = is coupling between layers, its not symmetric     # CHECKED

T = np.zeros((2 * R, 2 * R))

for i in np.arange(R, 2 * R):  # from R to 2R-1 meaning right nodes of the layer
    if i % 2 != 0:
        T[i - R, i] = t  # this is for S_down


# Identity matrix for placing H blocks

I_Q = np.eye(Q)


# Superdiagonal matrix for T blocks

S_up = np.diag(np.ones(Q - 1), k=1)


# Subdiagonal matrix for T blocks

S_down = np.diag(np.ones(Q - 1), k=-1)


# Construct the block matrix using Kronecker products

Hdevice1 = np.kron(I_Q, H) + np.kron(S_down, T) + np.kron(S_up, T.T)

np.save("H_R_leads_unit_cell.npy", Hdevice1)
