import numpy as np

from imageio import imread, imsave
from sklearn import metrics




def evaluate(change_map, true_mask):
    # 0  1
    matrix = metrics.confusion_matrix(true_mask.reshape(-1),
                                      change_map.reshape(-1))
    kappa = metrics.cohen_kappa_score(true_mask.reshape(-1),
                                      change_map.reshape(-1))
    TP = matrix[1, 1]
    TN = matrix[0, 0]
    FP = matrix[0, 1]
    FN = matrix[1, 0]
    Ra = (TP + TN) / (TP + TN + FP + FN)
    Rp = TP / (TP + FP) 
    Rm = FN / (TP + FN) 
    Rf = FP / (TP + FP) 
    kappa = kappa 
    print(f'TN:{TN}', f'TP:{TP}', f'FN:{FN}', f'FP:{FP}')
    print(f'Ra:{Ra:.4f}', f'Rp:{Rp:.4f}', f'Rm:{Rm:.4f}', f'Rf:{Rf:.4f}',
          f'kappa:{kappa:.4f}')
    return Ra,kappa


def display(changemap, true_mask):
    # 0  1
    Final_changemap = np.zeros((true_mask.shape[0], true_mask.shape[1], 3))
    for i in range(true_mask.shape[0]):
        for j in range(true_mask.shape[1]):
            if changemap[i, j] == 1 and true_mask[i, j] == 1:
                Final_changemap[i, j, 0] = 255
                Final_changemap[i, j, 1] = 255
                Final_changemap[i, j, 2] = 255
            elif changemap[i, j] == 0 and true_mask[i, j] == 1:
                Final_changemap[i, j, 0] = 255
            elif changemap[i, j] == 1 and true_mask[i, j] == 0:
                Final_changemap[i, j, 1] = 255
    return Final_changemap