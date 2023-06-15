import os
import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture


def parseOneLine(line: str) -> int:
    return float(line.split("\t")[1])


def contigLengthDist(bin_folder: str) -> List[int]:
    files = os.listdir(bin_folder)
    contigLenList = []
    for file in files:
        if file.split(".")[-1] == "fai":
            with open(os.path.join(bin_folder, file), "r") as rh:
                for oneLine in rh:
                    contigLenList.append(parseOneLine(oneLine.strip("\n")))
    return contigLenList


if __name__ == "__main__":
    contigLen = contigLengthDist("E:\\Data\\bins\\")
    contigLen = np.array(contigLen)[:, None]
    gmmModel = GaussianMixture(n_components=10, max_iter=1000)
    gmmModel.fit(contigLen)
    print(gmmModel.means_)
    wh = open("./GMM.pkl", "wb")
    pickle.dump(gmmModel, wh)
    wh.close()
    gmmLoad = pickle.load(open("./GMM.pkl", "rb"))
    print(gmmLoad.means_)
    print(gmmLoad.sample(100)[0])
    sortedCon = sorted(contigLen)[0:-2000]
    # the histogram of the data
    n, bins, patches = plt.hist(sortedCon, 100)
    plt.xlabel("Length")
    plt.ylabel("Probability")
    plt.show()
