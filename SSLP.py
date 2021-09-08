import os
import time

import cv2 as cv
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread, imsave
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

import CDTool


def num(array, element):
    count = 0
    for x in array:
        if x == element:
            count += 1
    return count


def visual(truth, x, y, size):
    true_mask = imread(truth)
    true_mask[true_mask > 120] = 255
    true_mask[true_mask != 255] = 0
    # true_mask[true_mask != 0] = 1
    plt.imshow(true_mask)
    rect = patches.Rectangle((y, x),
                             size[0],
                             size[0],
                             linewidth=0.5,
                             edgecolor='r',
                             facecolor='none')
    plt.subplot(1, 1, 1).add_patch(rect)
    plt.show()


def test(image1, image2, pixel, size):
    for (x, y) in pixel:
        visual1 = image1[x:x + size[0], y:y + size[1], :]
        visual2 = image2[x:x + size[0], y:y + size[1], :]
        plt.subplot(2, 2, 1)
        plt.imshow(image1)
        rect = patches.Rectangle((y, x),
                                 size[0],
                                 size[0],
                                 linewidth=0.5,
                                 edgecolor='r',
                                 facecolor='none')
        plt.subplot(2, 2, 1).add_patch(rect)
        plt.subplot(2, 2, 2)
        plt.imshow(image2)
        rect = patches.Rectangle((y, x),
                                 size[0],
                                 size[0],
                                 linewidth=0.5,
                                 edgecolor='b',
                                 facecolor='none')
        plt.subplot(2, 2, 2).add_patch(rect)
        plt.subplot(2, 2, 3)
        plt.imshow(visual1)
        plt.subplot(2, 2, 4)
        plt.imshow(visual2)
        plt.title(f'{x, y}')
        plt.show()


def creat_poor_data():
    if dataset == 'Sardinia':
        a = imread(os.path.join(root_path, 'data\Sardinia\Sardinia-sar.bmp'))
        b = imread(
            os.path.join(root_path, 'data\Sardinia\Sardinia-optical.bmp'))
        true_mask = imread(
            os.path.join(root_path, 'data\Sardinia\Sardinia-truth.bmp'))
        image_shape = (a.shape[0], a.shape[1])
        change_samples = np.zeros(image_shape)
        unchange_samples = np.zeros(image_shape)

        cp = [(105, 145), (88, 185), (82, 235), (82, 255)]
        ucp = [(110, 210), (80, 205), (135, 245), (150, 100), (200, 150),
               (50, 50), (150, 200), (50, 300)]
        c_size = (10, 10)
        uc_size = (10, 10)
    elif dataset == 'Shuguang_Village':
        a = imread(
            os.path.join(root_path,
                         'data\Shuguang Village\Shuguang Village-sar.jpg'))
        b = imread(
            os.path.join(root_path,
                         'data\Shuguang Village\Shuguang Village-optical.jpg'))
        true_mask = imread(
            os.path.join(root_path,
                         'data\Shuguang Village\Shuguang Village-truth.bmp'))
        image_shape = (a.shape[0], a.shape[1])
        change_samples = np.zeros(image_shape)
        unchange_samples = np.zeros(image_shape)

        cp = [(75, 100), (30, 165), (15, 175), (40, 120), (35, 100)]
        ucp = [(60, 210), (100, 50), (210, 75), (260, 225), (250, 320),
               (150, 325), (50, 325), (100, 400), (250, 370), (18, 150)]
        unchange_samples[18:18 + 5, 150:150 + 5] = np.ones((5, 5))
        c_size = (10, 10)
        uc_size = (10, 10)
    elif dataset == 'Yellow_River':
        a = imread(
            os.path.join(root_path, 'data\Yellow River\Yellow River-sar.jpg'))
        b = imread(
            os.path.join(root_path,
                         'data\Yellow River\Yellow River-optical.jpg'))
        true_mask = imread(
            os.path.join(root_path,
                         'data\Yellow River\Yellow River-truth.bmp'))
        image_shape = (a.shape[0], a.shape[1])
        change_samples = np.zeros(image_shape)
        unchange_samples = np.zeros(image_shape)

        cp = [(120, 115), (150, 117), (250, 158), (280, 125), (302, 10),
              (95, 138), (295, 90), (172, 128)]
        ucp = [(320, 20), (310, 120), (330, 190), (200, 120), (150, 250),
               (40, 80), (80, 190), (130, 150), (220, 80), (100, 100)]
        c_size = (5, 5)
        uc_size = (10, 10)

    # test(a, b, cpoint, c_size)
    #visual(true_mask, 150, 117, (5,5))  # test
    for i in range(len(cp)):
        change_samples[cp[i][0]:cp[i][0] + c_size[0],
                       cp[i][1]:cp[i][1] + c_size[1]] = np.ones(c_size)
    for j in range(len(ucp)):
        unchange_samples[ucp[j][0]:ucp[j][0] + uc_size[0],
                         ucp[j][1]:ucp[j][1] + uc_size[1]] = np.ones(uc_size)

    np.save(os.path.join(root_path, f'{dataset}_change_poor_label.npy'),
            change_samples)
    np.save(os.path.join(root_path, f'{dataset}_unchange_poor_label.npy'),
            unchange_samples)

    return a, b, true_mask, image_shape, change_samples, unchange_samples


def kmeans_cluster(image, image_shape, k):
    t1 = time.process_time()
    x_train = image.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k).fit(x_train)
    predict = kmeans.predict(x_train)
    predict = predict.reshape(image_shape)
    t2 = time.process_time()
    #print("time:".format(t2 - t1))
    return predict


def propagation(change_samples, unchange_samples, predict1, predict2, image1,
                image2):
    change_label1 = []
    unchange_label1 = []
    for (i, j) in np.argwhere(change_samples == 1):
        change_label1.append(predict1[i, j])
    for (i, j) in np.argwhere(unchange_samples == 1):
        unchange_label1.append(predict1[i, j])
    repetitive_label1 = [
        x for x in list(set(change_label1)) if x in list(set(unchange_label1))
    ]
    print('conflict class label 1:', repetitive_label1)
    change_label2 = []
    unchange_label2 = []
    for (i, j) in np.argwhere(change_samples == 1):
        change_label2.append(predict2[i, j])
    for (i, j) in np.argwhere(unchange_samples == 1):
        unchange_label2.append(predict2[i, j])
    repetitive_label2 = [
        x for x in list(set(change_label2)) if x in list(set(unchange_label2))
    ]
    print('conflict class label 2:', repetitive_label2)

    for i in tqdm(range(len(repetitive_label1))):
        cluster_samples = np.zeros_like(predict1)
        alpha = np.max(predict1)
        for (x, y) in np.argwhere(predict1 == repetitive_label1[i]):
            cluster_samples[x, y] = 1
        cluster_center = np.logical_and(
            cluster_samples,
            np.logical_or(change_samples,
                          unchange_samples).astype(int)).astype(int)
        num_cluster = num(cluster_center.reshape(-1, 1), 1)
        for (x, y) in np.argwhere(cluster_samples == 1):
            cc = np.zeros((num_cluster, 3))
            t = 0
            for (u, v) in np.argwhere(cluster_center == 1):
                cc[t, :] = image1[u, v, :]
                t += 1
            dist_matrix = np.sum(np.asarray(image1[x, y, :] - cc)**2, axis=1)

            predict1[x, y] = alpha + np.argmin(dist_matrix) + 1
    for i in tqdm(range(len(repetitive_label2))):
        cluster_samples = np.zeros_like(predict2)
        alpha = np.max(predict2)
        for (x, y) in np.argwhere(predict2 == repetitive_label2[i]):
            cluster_samples[x, y] = 1
        cluster_center = np.logical_and(
            cluster_samples,
            np.logical_or(change_samples,
                          unchange_samples).astype(int)).astype(int)
        num_cluster = num(cluster_center.reshape(-1, 1), 1)
        for (x, y) in np.argwhere(cluster_samples == 1):
            cc = np.zeros((num_cluster, 3))
            t = 0
            for (u, v) in np.argwhere(cluster_center == 1):
                cc[t, :] = image2[u, v, :]
                t += 1
            dist_matrix = np.sum(np.asarray(image2[x, y, :] - cc)**2, axis=1)

            predict2[x, y] = alpha + np.argmin(dist_matrix) + 1

    change = np.zeros_like(change_samples)
    unchange = np.zeros_like(unchange_samples)
    for (i, j) in tqdm(np.argwhere(change_samples == 1)):
        change1 = predict1.copy()
        change2 = predict2.copy()
        change1[change1 == predict1[i, j]] = 10000
        change1[change1 != 10000] = 0
        change1[change1 != 0] = 1
        change2[change2 == predict2[i, j]] = 10000
        change2[change2 != 10000] = 0
        change2[change2 != 0] = 1
        c = np.logical_and(change1, change2).astype(int)
        change += c
    for (i, j) in tqdm(np.argwhere(unchange_samples == 1)):
        unchange1 = predict1.copy()
        unchange2 = predict2.copy()
        unchange1[unchange1 == predict1[i, j]] = 10000
        unchange1[unchange1 != 10000] = 0
        unchange1[unchange1 != 0] = 1
        unchange2[unchange2 == predict2[i, j]] = 10000
        unchange2[unchange2 != 10000] = 0
        unchange2[unchange2 != 0] = 1
        uc = np.logical_and(unchange1, unchange2).astype(int)
        unchange += uc

    change[change != 0] = 1
    unchange[unchange != 0] = 1

    return change, unchange


def eliminate_density(change, unchange, change_sign, unchange_sign, beta_c,
                      beta_uc):
    index_change_transmit = np.argwhere((change - change_sign) == 1)
    index_unchange_transmit = np.argwhere((unchange - unchange_sign) == 1)

    change_transmit_dist = np.zeros_like(change)
    for (x, y) in tqdm(index_change_transmit):
        distance_matrix = np.sqrt(
            np.sum(np.asarray(np.array([x, y]) - np.argwhere(change == 1))**2,
                   axis=1))
        distance_matrix = np.sort(distance_matrix)
        change_transmit_dist[x, y] = (np.sum(distance_matrix[0:5])) / 5
    change_list = list(change_transmit_dist.reshape(-1))
    change_list = list(set(change_list))
    change_list.remove(0)
    change_list.sort()
    threshold_c = change_list[int(len(change_list) * beta_c)]
    change_transmit_dist[change_transmit_dist > threshold_c] = 0
    change_transmit_dist[change_transmit_dist != 0] = 1
    change = change_transmit_dist + change_sign
    change[change != 0] = 1

    unchange_transmit_dist = np.zeros_like(unchange)
    for (x, y) in tqdm(index_unchange_transmit):
        distance_matrix = np.sqrt(
            np.sum(np.asarray(np.array([x, y]) -
                              np.argwhere(unchange == 1))**2,
                   axis=1))
        distance_matrix = np.sort(distance_matrix)
        unchange_transmit_dist[x, y] = (np.sum(distance_matrix[0:5])) / 5
    unchange_list = list(unchange_transmit_dist.reshape(-1))
    unchange_list = list(set(unchange_list))
    unchange_list.remove(0)
    unchange_list.sort()
    threshold_uc = unchange_list[int(len(unchange_list) * beta_uc)]
    unchange_transmit_dist[unchange_transmit_dist > threshold_uc] = 0
    unchange_transmit_dist[unchange_transmit_dist != 0] = 1
    unchange = unchange_transmit_dist + unchange_sign
    unchange[unchange != 0] = 1

    return change, unchange


def expand(change, unchange, block_size):
    n = int((block_size - 1) / 2)
    change1 = change.copy()
    unchange1 = unchange.copy()
    for (i, j) in np.argwhere(change == 1):
        if change1[i - n: i + block_size - n, j - n:j + block_size - n].shape[0] * \
                change1[i - n: i + block_size - n, j - n:j + block_size - n].shape[1] == block_size * block_size:
            change1[i - n:i + block_size - n,
                    j - n:j + block_size - n] = np.ones(
                        (block_size, block_size))
    for (i, j) in np.argwhere(unchange == 1):
        if unchange1[i - n: i + block_size - n, j - n:j + block_size - n].shape[0] * \
                unchange1[i - n: i + block_size - n, j - n:j + block_size - n].shape[1] == block_size * block_size:
            unchange1[i - n:i + block_size - n,
                      j - n:j + block_size - n] = np.ones(
                          (block_size, block_size))
    for (i, j) in np.argwhere(change == 1):
        if np.sum(unchange1[i - n:i + block_size - n,
                            j - n:j + block_size - n]) != 0:
            change[i, j] = 0
    for (i, j) in np.argwhere(unchange == 1):
        if np.sum(change1[i - n:i + block_size - n,
                          j - n:j + block_size - n]) != 0:
            unchange[i, j] = 0
    change2 = change.copy()
    unchange2 = unchange.copy()
    for (i, j) in np.argwhere(change == 1):
        if change2[i - n: i + block_size - n, j - n:j + block_size - n].shape[0] * \
                change2[i - n: i + block_size - n, j - n:j + block_size - n].shape[1] == block_size * block_size:
            change2[i - n:i + block_size - n,
                    j - n:j + block_size - n] = np.ones(
                        (block_size, block_size))
    for (i, j) in np.argwhere(unchange == 1):
        if unchange2[i - n: i + block_size - n, j - n:j + block_size - n].shape[0] * \
                unchange2[i - n: i + block_size - n, j - n:j + block_size - n].shape[1] == block_size * block_size:
            unchange2[i - n:i + block_size - n,
                      j - n:j + block_size - n] = np.ones(
                          (block_size, block_size))

    return change1, unchange1


# classifier
def rf_classifier(Change, UnChange, im1, im2):
    index_Change = np.where(Change.reshape(-1, 1) == 1)[0]
    index_unChange = np.where(UnChange.reshape(-1, 1) == 1)[0]
    image1 = im1.reshape(-1, 3)
    image2 = im2.reshape(-1, 3)
    image1_train = np.concatenate(
        (image1[index_Change], image1[index_unChange]), axis=0)
    image2_train = np.concatenate(
        (image2[index_Change], image2[index_unChange]), axis=0)
    x_train = np.concatenate((image1_train, image2_train), axis=1)
    y_train = np.zeros(x_train.shape[0])
    for i in range(len(index_Change)):
        y_train[i] = 1

    rfc = RandomForestClassifier(random_state=0)
    rfc.fit(x_train, y_train)

    x_test = np.concatenate((image1, image2), axis=1)
    change_map = rfc.predict(x_test)

    return change_map


def knn_classifier(Change, UnChange, im1, im2, n):
    index_Change = np.where(Change.reshape(-1, 1) == 1)[0]
    index_unChange = np.where(UnChange.reshape(-1, 1) == 1)[0]
    image1 = im1.reshape(-1, 3)
    image2 = im2.reshape(-1, 3)
    image1_train = np.concatenate(
        (image1[index_Change], image1[index_unChange]), axis=0)
    image2_train = np.concatenate(
        (image2[index_Change], image2[index_unChange]), axis=0)
    x_train = np.concatenate((image1_train, image2_train), axis=1)
    y_train = np.zeros(x_train.shape[0])
    for i in range(len(index_Change)):
        y_train[i] = 1

    knn = KNeighborsClassifier(n_neighbors=n, algorithm='auto')  # knn
    knn.fit(x_train, y_train)

    x_test = np.concatenate((image1, image2), axis=1)
    change_map = knn.predict(x_test)

    return change_map


def sslp(k=80, beta_c=0.8, beta_uc=0.8):

    # 1. load data
    print('load data start... ...')
    global image_shape
    Image1, Image2, true_mask, image_shape, change_samples, unchange_samples = creat_poor_data(
    )
    change_samples = np.load(
        os.path.join(root_path, f'{dataset}_change_poor_label.npy'))
    unchange_samples = np.load(
        os.path.join(root_path, f'{dataset}_unchange_poor_label.npy'))
    # 2. clp
    print('cluster start... ...')
    predict1 = kmeans_cluster(Image1, image_shape, k)
    predict2 = kmeans_cluster(Image2, image_shape, k)

    print('Propagation start... ...')
    change, unchange = propagation(change_samples, unchange_samples, predict1,
                                   predict2, Image1, Image2)
    # denoising
    change, unchange = eliminate_density(change, unchange, change_samples,
                                         unchange_samples, beta_c, beta_uc)

    # expand
    change, unchange = expand(change, unchange, block_size=3)
    for (i, j) in np.argwhere(np.logical_and(change, unchange) == 1):
        change[i, j] = 0
        unchange[i, j] = 0

    # 3. classifier
    print('classify start... ...')
    #binary_image = rf_classifier(change, unchange, Image1, Image2).reshape(image_shape)
    predict = knn_classifier(change, unchange, Image1, Image2,
                             n=7).reshape(image_shape).astype('uint8')
    binary_image = cv.medianBlur(predict, 3)

    true_mask[true_mask < 120] = 0
    true_mask[true_mask != 0] = 1
    auc, kappa = CDTool.evaluate(binary_image, true_mask)

    binary_image[binary_image == 1] = 255
    imsave(os.path.join(root_path, f'binary_image_{kappa:.4f}.bmp'),
           binary_image)


if __name__ == "__main__":
    global root_path, dataset
    root_path = os.path.abspath(
        os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".")
    dataset = 'Sardinia'  # Sardinia, Shuguang_Village, Yellow_River
    print('root_path:', root_path)
    sslp(k=80, beta_c=0.8, beta_uc=0.8)
