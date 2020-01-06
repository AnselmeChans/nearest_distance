# =========================IMPORT PACKAGES ==============================
import random
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# ============================= Functions ================================

def get_point_nearest(node, list_coordinate): 
    list_cordinate_nodes = np.asarray(list_coordinate)
    deltas_x_y = list_cordinate_nodes - node
    dist_node = np.einsum('ij, ij -> i', deltas_x_y, deltas_x_y)
    return np.argmin(dist_node)
    
    
def closest_node(node, nodes):
    return nodes[cdist([node], nodes).argmin()]


def get_dist_euclidienn(coords): 
    dist_eucl = cdist(coords, coords, 'euclidean')
    return dist_eucl


def closest(df):
    X = df[['x', 'y']]
    print("================= X ========================")
    print(X)
    dist = cdist(X, X)
    print(dist)
    v = np.argsort(dist)
    print("================V====================")
    print(v)
    print("v[:, 0] ====+> {}".format(v[:, 0]))
    print("EUCLIDIAN DISTANCE ======================")
    print(dist[v[:, 0], v[:, 1]])
    return df.assign(euclidean_distance=dist[v[:, 0], v[:, 1]],
                        nearest_neighbour=df.distinction.values[v[:, 1]])


# df.groupby('time').apply(closest).reset_index(drop=True)

#========================= Main principal ====================================
if __name__ == '__main__':
    list_coordinate = []
    for i in range(10): 
        x = random.randrange(0, 1600)
        y = random.randrange(0, 900)
        list_coordinate.append((x, y))
    print(list_coordinate)


    list_x = random.sample(range(0, 1600), 10)
    list_y = random.sample(range(0, 900), 10)
    time = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4]
    distnction = ["pix1", "pix2", "pix3", "pix4", "pix5", "pix6", "pix7", "pix8", "pix9", "pix10"]
    print(list_x)
    print(list_y)
    print(time)
    print(distnction)
    df = pd.DataFrame({'x': list_x, 'y': list_y, 'distinction': distnction})
    print("-----------------------")

    a = closest(df)
    print("================CLOSEST================")
    print(a)
    print(type(a))
