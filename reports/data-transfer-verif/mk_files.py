import numpy as np
import os

transfered = np.loadtxt("ls_mk_files.txt").tolist()
to_transfer = np.loadtxt("train_MK.txt").tolist()

diff = [x for x in to_transfer if x not in transfered]

with open("diff.txt", "w") as output:
    for img in diff:
        output.write(img + "\n")
