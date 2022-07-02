"""
人工智能课设
201942852022年6月22日17:43:21
杨振宇 自动化1904
单个图片的SVM检测
"""
import numpy as np
import cv2
import pickle
import joblib
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import os

label2id = {'合格':0, '鼓包缺陷':1, '弯曲缺陷':2}


def extract_sift_features(X):
    image_descriptors = []
    sift = cv2.SIFT_create()

    for i in range(len(X)):
        kp, des = sift.detectAndCompute(X[i], None)
        image_descriptors.append(des)

    return image_descriptors

def kmeans_bow(all_descriptors, num_clusters):
    bow_dict = []
    kmeans = KMeans(n_clusters=num_clusters).fit(all_descriptors)
    bow_dict = kmeans.cluster_centers_
    return bow_dict
num_clusters = 100
BoW = pickle.load(open('bow_dictionary150.pkl', 'rb'))
def create_features_bow(image_descriptors, BoW, num_clusters):
    X_features = []
    for i in range(len(image_descriptors)):
        features = np.array([0] * num_clusters)

        if image_descriptors[i] is not None:
            distance = cdist(image_descriptors[i], BoW)
            argmin = np.argmin(distance, axis=1)
            for j in argmin:
                features[j] += 1
        X_features.append(features)
    return X_features

clf = joblib.load("my_model.m")


#img_test = cv2.imread('huotui/test/hege/988.png')#合格图片测试
img_test = cv2.imread('huotui/test/gubao/986.png')#鼓包缺陷测试
#img_test = cv2.imread('huotui/test/wanqu/1044.png')#弯曲缺陷测试

img = [img_test]
img_sift_feature = extract_sift_features(img)
img_bow_feature = create_features_bow(img_sift_feature, BoW, num_clusters)
img_predict=clf.predict(img_bow_feature)
print(img_predict)
for key, value in label2id.items():
    if value == img_predict[0]:
        print('Your prediction: ', key)
