import os
import numpy as np
import cv2
import pickle
import sklearn
import joblib
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split

#数据集加载
def read_data(label2id):
    X = []
    Y = []
    for label in os.listdir('huotui/data'):
        for img_file in os.listdir(os.path.join('huotui/data', label)):
            img = cv2.imread(os.path.join('huotui/data', label, img_file))
            X.append(img)
            Y.append(label2id[label])
    return X, Y


#为数据集加标签
label2id = {'hege':0, 'gubao':1, 'wanqu':2}
X, Y = read_data(label2id)


#图片sift特征值的描述子提取
def extract_sift_features(X):
    image_descriptors = []
    sift = cv2.SIFT_create()

    for i in range(len(X)):
        kp, des = sift.detectAndCompute(X[i], None)
        image_descriptors.append(des)

    return image_descriptors

image_descriptors = extract_sift_features(X)


#kmans、词袋模型
all_descriptors = []
for descriptors in image_descriptors:
    if descriptors is not None:
        for des in descriptors:
            all_descriptors.append(des)
#kmeans参数模型，100 个聚类构建 词袋，每个图像将被矢量化为100维矢量
def kmeans_bow(all_descriptors, num_clusters):
    bow_dict = []
    kmeans = KMeans(n_clusters=num_clusters).fit(all_descriptors)
    bow_dict = kmeans.cluster_centers_
    return bow_dict

num_clusters = 100
#生成词袋模型
if not os.path.isfile('bow_dictionary150.pkl'):
    BoW = kmeans_bow(all_descriptors, num_clusters)
    pickle.dump(BoW, open('bow_dictionary150.pkl', 'wb'))
else:
    BoW = pickle.load(open('bow_dictionary150.pkl', 'rb'))


#返回基于词袋的描述符提取器计算得到的描述符，并用数组来存储
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

X_features = create_features_bow(image_descriptors, BoW, num_clusters)


#数据集划分
X_train = []
X_test = []
Y_train = []
Y_test = []
X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y, test_size=0.2, random_state=42)#以训练集的20%做验证集验证效果

svm = sklearn.svm.SVC(C = 10)
svm.fit(X_train, Y_train)
joblib.dump(svm, "my_model.m")

#准确率
print('Your accuracy: {:.2%}'.format(svm.score(X_test, Y_test)))
#print(svm.support_vectors_)
#训练集中选取一张验证，测试效果
img_test = cv2.imread('huotui/test/hege/985.png')
img = [img_test]
img_sift_feature = extract_sift_features(img)
img_bow_feature = create_features_bow(img_sift_feature, BoW, num_clusters)
img_predict = svm.predict(img_bow_feature)

print(img_predict)
for key, value in label2id.items():
    if value == img_predict[0]:
        print('Your prediction: ', key)

#图片展示
cv2.imshow("Img", img_test)
cv2.waitKey(0)
