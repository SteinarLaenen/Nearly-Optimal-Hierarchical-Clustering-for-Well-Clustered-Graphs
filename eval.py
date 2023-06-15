"""
This script evaluates the performance of the Algorithm SpecWRSC against  other state-of-the-art
algorithms for hierarchical clustering.
"""

import test
import datetime
import time
import sys

def main():
    sys.setrecursionlimit(10000)

    filename = str("./results/Results_" + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H_%M_%S') + ".txt")
    out = open(filename, "w")

    # The type of trees to be tested
    tree_types = [
                "average_linkage",
                "Prune_Merge",
                "spectral_degree_clustering"
                 ]

    # run the tests
    test.test_stochastic_block_model(tree_types, out)
    test.test_hierarchical_stochastic_block_model(tree_types, out)
    test.test_real_data(tree_types, out)
    out.close()

if __name__ == "__main__":
    main()
