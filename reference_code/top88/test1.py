import numpy as np

from scipy.sparse import coo_matrix

nelx = 10
nely = 5
rmin = 1.5
# Filter: Build (and assemble) the index+data vectors for the coo matrix format
nfilter = int(nelx * nely * ((2 * (np.ceil(rmin) - 1) + 1) ** 2))
iH = np.zeros(nfilter)
jH = np.zeros(nfilter)
sH = np.zeros(nfilter)
cc = 0
for i in range(nelx):
    for j in range(nely):
        row = i * nely + j
        kk1 = int(np.maximum(i - (np.ceil(rmin) - 1), 0))
        kk2 = int(np.minimum(i + np.ceil(rmin), nelx))
        ll1 = int(np.maximum(j - (np.ceil(rmin) - 1), 0))
        ll2 = int(np.minimum(j + np.ceil(rmin), nely))
        for k in range(kk1, kk2):
            for l in range(ll1, ll2):
                col = k * nely + l
                fac = rmin - np.sqrt(((i-k) * (i-k) + (j-l) * (j-l)))
                iH[cc] = row
                jH[cc] = col
                sH[cc] = np.maximum(0.0, fac)
                cc = cc + 1
# Finalize assembly and convert to csc format
H = coo_matrix((sH, (iH, jH)), shape=(nelx*nely, nelx*nely)).tocsc()
Hs = H.sum(1)
print("hhhhhhhhhhhhhhhhhhhhhh")