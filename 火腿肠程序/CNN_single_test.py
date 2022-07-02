'''
人工智能课设
2022年6月22日19:50:09
2094285杨振宇自动化1904
cnn单个图片检测
'''
import numpy as np
from tensorflow import keras
from PIL import Image

model = keras.models.load_model('htcmodel.h5')
label2id = {'合格':0, '弯曲缺陷':1, '鼓包缺陷':2}
# predicting images
im =Image.open("huotui/test/gubao/1026.png")
im = im.resize((150, 150))
im = np.array(im)
x = np.expand_dims(im, axis=0)
images = np.vstack([x])
ret = model.predict(images)
number = np.argmax(ret)
print(number)
for key, value in label2id.items():
    if value == number:
        print('你的预测结果是: ', key)