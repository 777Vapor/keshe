import numpy as np
from tensorflow import keras
from PIL import Image
import os

def read_data():
    X = []
    Y = []
    for filepath, dirnames, filenames in os.walk(r'F:\qq文件\剪刀石头布\test'):
        for filename in filenames:
         img=os.path.join(filepath, filename)

         X.append(img)
         Y.append(filename)
    return X, Y
X,Y=read_data()
model = keras.models.load_model('rpsmodel.h5')
label2id = {'布':0, '石头':1, '剪刀':2}
# predicting images
for i in range(len(X)):
    im = Image.open(X[i]).convert("RGB")
    im = im.resize((150, 150))
    im = np.array(im)

    x = np.expand_dims(im, axis=0)
    images = x
    ret = model.predict(images)

    number = np.argmax(ret)
    print("图片名称为")
    print(Y[i])
    for key, value in label2id.items():
        if value == number:
            print('你的预测结果是: ', key)
            print("\n")

#from keras.utils import plot_model
#plot_model(model, to_file='DResLayer_model.png', show_shapes=True)  # 绘制并保存模型结构图
