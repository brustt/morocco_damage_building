"""
Slow feature analysis
C. Wu, B. Du, and L. Zhang, “Slow feature analysis for change detection in multispectral imagery,” IEEE Trans. Geosci. Remote Sens., vol. 52, no. 5, pp. 2858–2874, 2014.

Implementation from : https://github.com/ChenHongruixuan/ChangeDetectionRepository/blob/master/Methodology/Traditional/SFA/isfa.py
"""

import cv2
import numpy as np
from scipy.linalg import eig
from scipy.stats import chi2
from sklearn.cluster import KMeans
from src.io.utils_io import check_dir

from src.modelling.utils import otsu
import time
import imageio
from src.config import *
from pathlib import Path
import rioxarray as rioxr
import pandas as pd

class ISFA(object):
    def __init__(self, img_X, img_Y, data_format='CHW'):
        """
        the init function
        :param img_X: former temporal image, its dim is (band_count, width, height)
        :param img_Y: latter temporal image, its dim is (band_count, width, height)
        """
        if data_format == 'HWC':
            self.img_X = np.transpose(img_X, [2, 0, 1])
            self.img_Y = np.transpose(img_Y, [2, 0, 1])
        else:
            self.img_X = img_X
            self.img_Y = img_Y

        channel, height, width = self.img_X.shape
        self.L = np.zeros((channel - 2, channel))  # (C-2, C)
        for i in range(channel - 2):
            self.L[i, i] = 1
            self.L[i, i + 1] = -2
            self.L[i, i + 2] = 1
        self.Omega = np.dot(self.L.T, self.L)  # (C, C)
        self.norm_method = ['LSR', 'NR', 'OR']

    def isfa(self, max_iter=30, epsilon=1e-6, norm_trans=False, regular=False):
        """
         extract change and unchange info of temporal images based on USFA
         if max_iter == 1, ISFA is equal to SFA
        :param max_iter: the maximum count of iteration
        :param epsilon: convergence threshold
        :param norm_trans: whether normalize the transformation matrix
        :return:
            ISFA_variable: ISFA variable, its dim is (band_count, width * height)
            lamb: last lambda
            all_lambda: all lambda in convergence process
            trans_mat: transformation matrix
            T: last IWD, if max_iter == 1, T is chi-square distance
            weight: the unchanged probability of each pixel
        """

        bands_count, img_height, img_width = self.img_X.shape
        P = img_height * img_width
        # row-major order after reshape
        img_X = np.reshape(self.img_X, (-1, img_height * img_width))  # (band, width * height)
        img_Y = np.reshape(self.img_Y, (-1, img_height * img_width))  # (band, width * height)
        lamb = 100 * np.ones((bands_count, 1))
        all_lambda = []
        weight = np.ones((img_width, img_height))  # (1, width * height)
        # weight[302:343, 471] = 1  # init seed
        # weight[209, 231:250] = 1
        # weight[335:362, 570] = 1
        # weight[779, 332:387] = 1

        weight = np.reshape(weight, (-1, img_width * img_height))
        for _iter in range(max_iter):
            sum_w = np.sum(weight)
            mean_X = np.sum(weight * img_X, axis=1, keepdims=True) / np.sum(weight)  # (band, 1)
            mean_Y = np.sum(weight * img_Y, axis=1, keepdims=True) / np.sum(weight)  # (band, 1)
            center_X = (img_X - mean_X)
            center_Y = (img_Y - mean_Y)

            # cov_XY = covw(center_X, center_Y, weight)  # (2 * band, 2 * band)
            # cov_X = cov_XY[0:bands_count, 0:bands_count]
            # cov_Y = cov_XY[bands_count:2 * bands_count, bands_count:2 * bands_count]
            var_X = np.sum(weight * np.power(center_X, 2), axis=1, keepdims=True) / ((P - 1) * sum_w / P)
            var_Y = np.sum(weight * np.power(center_Y, 2), axis=1, keepdims=True) / ((P - 1) * sum_w / P)
            std_X = np.reshape(np.sqrt(var_X), (bands_count, 1))
            std_Y = np.reshape(np.sqrt(var_Y), (bands_count, 1))

            # normalize image
            norm_X = center_X / std_X
            norm_Y = center_Y / std_Y
            diff_img = (norm_X - norm_Y)
            mat_A = np.dot(weight * diff_img, diff_img.T) / ((P - 1) * sum_w / P)
            mat_B = (np.dot(weight * norm_X, norm_X.T) +
                     np.dot(weight * norm_Y, norm_Y.T)) / (2 * (P - 1) * sum_w / P)
            if regular:
                penalty = np.trace(mat_B) / np.trace(self.Omega)
                mat_B += penalty * self.Omega
            # solve generalized eigenvalue problem and get eigenvalues and eigenvector
            eigenvalue, eigenvector = eig(mat_A, mat_B)
            eigenvalue = eigenvalue.real  # discard imaginary part
            idx = eigenvalue.argsort()
            eigenvalue = eigenvalue[idx]

            # make sure the max absolute value of vector is 1,
            # and the final result will be more closer to the matlab result
            aux = np.reshape(np.abs(eigenvector).max(axis=0), (1, bands_count))
            eigenvector = eigenvector / aux

            # print sqrt(lambda)
            if (_iter + 1) == 1:
                print('sqrt lambda:')
            print(np.sqrt(eigenvalue))

            eigenvalue = np.reshape(eigenvalue, (bands_count, 1))  # (band, 1)
            threshold = np.max(np.abs(np.sqrt(lamb) - np.sqrt(eigenvalue)))
            # if sqrt(lambda) converge
            if threshold < epsilon:
                break
            lamb = eigenvalue
            all_lambda = lamb if (_iter + 1) == 1 else np.concatenate((all_lambda, lamb), axis=1)
            # the order of the slowest features is determined by the order of the eigenvalues
            trans_mat = eigenvector[:, idx]
            # satisfy the constraints(3)
            if norm_trans:
                output_signal_std = 1 / np.sqrt(np.diag(np.dot(trans_mat.T, np.dot(mat_B, trans_mat))))
                trans_mat = output_signal_std * trans_mat
            ISFA_variable = np.dot(trans_mat.T, norm_X) - np.dot(trans_mat.T, norm_Y)

            if (_iter + 1) == 1:
                T = np.sum(np.square(ISFA_variable) / np.sqrt(lamb), axis=0, keepdims=True)  # chi square
            else:
                T = np.sum(np.square(ISFA_variable) / np.sqrt(lamb), axis=0, keepdims=True)  # IWD
            weight = 1 - chi2.cdf(T, bands_count)

        if (_iter + 1) == max_iter:
            print('the lambda may not be converged')
        else:
            print('the lambda is converged, the iteration is %d' % (_iter + 1))

        return ISFA_variable, lamb, all_lambda, trans_mat, T, weight


def main():
    
    type_item = "visual"
    town_test = "Talat_Nyaaqoub"
    s_buffer = 200
    out_dir = check_dir(results_dir_path, "sfa", town_test)
    
    registration_path = Path(processed_dir_path, type_item, town_test)
    #post_path = Path(registration_path, f"after_{s_buffer}_shifted__global_10300500E4F91500-visual.tif")
    #pre_path = Path(registration_path, f"before_{s_buffer}_104001004F028F00-visual.tif")
    pre_path = Path(interim_dir_path, type_item, town_test, "nearest","before", f"{s_buffer}_1040010077691A00-visual.tif")
    #post_path = Path(registration_path, "registration", f"after_{s_buffer}_shifted__global_10300500E4F92300-visual.tif")
    post_path = Path(registration_path, f"{s_buffer}_after_10300500E4F92300-visual_rgsted.tif")

    pre_img = rioxr.open_rasterio(pre_path).data
    post_img = rioxr.open_rasterio(post_path).data
    
    pre_img = pre_img.transpose(1, 2, 0)
    post_img = post_img.transpose(1, 2, 0)
    
    #pre_img = cv2.resize(pre_img, (512, 512))
    #post_img = cv2.resize(post_img, (512, 512))
    
    pre_img = cv2.GaussianBlur(pre_img, (15,15),0)
    post_img = cv2.GaussianBlur(post_img, (15,15),0)


    pre_img = pre_img.transpose(2, 0, 1)
    post_img = post_img.transpose(2, 0, 1)
    
    channel, img_height, img_width = pre_img.shape
    
    tic = time.time()
    sfa = ISFA(pre_img, post_img)
    # when max_iter is set to 1, ISFA becomes SFA
    bn_SFA_variable, bn_lamb, bn_all_lambda, bn_trans_mat, bn_iwd, bn_isfa_w = sfa.isfa(max_iter=50, epsilon=1e-3,
                                                                                        norm_trans=True)
    sqrt_chi2 = np.sqrt(bn_iwd)
    bcm = np.ones((1, img_height * img_width))
    thre = otsu(sqrt_chi2)
    bcm[sqrt_chi2 > thre] = 255
    bcm = np.reshape(bcm, (img_height, img_width))
    
    #bcm = cv2.morphologyEx(bcm.astype(np.uint8), cv2.MORPH_OPEN, (100, 100))

    out_path = os.path.join(out_dir, f'{s_buffer}_{town_test}.png')
    imageio.imwrite(out_path, bcm.astype(np.uint8))
    bcm = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((10,10),np.uint8)
    #bcm = cv2.erode(bcm, kernel)
    bcm = cv2.morphologyEx(bcm, cv2.MORPH_OPEN, kernel)

    out_path = os.path.join(out_dir, f'open_{kernel.shape[0]}_{s_buffer}_{town_test}.png')
    imageio.imwrite(out_path, bcm.astype(np.uint8))

    toc = time.time()
    print(toc - tic)


if __name__ == '__main__':
    main()