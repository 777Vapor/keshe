'''
人工智能课设
2022年6月22日17:44:21
20194285杨振宇自动化1904
图片分割与svm检测
'''
import os
import numpy as np
import cv2
import pickle
import joblib
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# 二值化颜色翻转与开闭运算处理
label2id = {'合格':0, '鼓包缺陷':1, '弯曲缺陷':2}
def change(img):

    img = cv2.GaussianBlur(img, (3, 3), 1.3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((23, 23), np.uint8)
    # 图像开运算和闭运算
    result = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,kernel)#029开头白底图片进行开运算
    #result = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)#089开头黑底图片进行开运算
    result = 255 - result
    ret, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("Fo",result)
    return result
#获取轮廓信息
def getRoi(im_out, img):
    '''
    im_out: 预处理好的二值图
    img : 原图
    '''
    #获取图片的位置，大小等各项参数
    num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(im_out, connectivity=8, ltype=cv2.CV_32S)
    image = np.copy(img)
    a=0
    roi_list = []
    for t in range(1, num_labels, 1):
        a=a+1
        x, y, w, h, area = stats[t]
        if area<6000 :
            a = a - 1
            continue
        # 画出外接矩形
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2, 8, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, str(a), (x, y), font, 2, (0, 255, 0), 3)
        # 保存roi的坐标和长宽
        roi_list.append((x, y, w, h))

    return num_labels, labels, image, roi_list


# 保存感兴趣的区域
def saveRoi(src, roi_list):
    '''
    src: 原图的copy
    roi_list: List,保存的roi位置信息
    '''
    for i in range(len(roi_list)):
        x, y, w, h = roi_list[i]
        roi = src[y:y+h, x:x+w]
        cv2.imwrite("roi_%d.jpg"%i, roi)
        print("No.%02d Finished! "%i)


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

#主函数
if __name__ == '__main__':
    # 预处理
    img = cv2.imread("huotui/yanshou/029025.bmp");
    #img = cv2.imread("huotui/yanshou/089744.bmp");
    img = cv2.resize(img, (640, 640))
    # 调用调整函数
    im_out = change(img)

    # 利用连通器寻找到需要提取的 roi
    num_labels, labels, image, roi_list = getRoi(im_out, img)
    clf = joblib.load("my_model.m")
    # 保存roi
    saveRoi(img, roi_list)
    a = 0
    for i in range(len(roi_list)):
        path="roi_%d.jpg"%i
        img_test = cv2.imread(path)
        img = [img_test]
        img_sift_feature = extract_sift_features(img)
        img_bow_feature = create_features_bow(img_sift_feature, BoW, num_clusters)
        img_predict = clf.predict(img_bow_feature)
        print(img_predict)
        for key, value in label2id.items():
            if value == img_predict[0]:
                print('Your prediction: ', key)
    print( end=':')
    for i in range(len(roi_list)):
        path = "roi_%d.jpg" % i
        os.remove(path)

    # Display images.
    cv2.imshow("Foreground", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(a)




