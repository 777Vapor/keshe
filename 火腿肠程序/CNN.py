#20194285 杨振宇 自动化1904
#人工智能课程设计
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
# 数据增强器
training_datagen = ImageDataGenerator(

    rescale=1. / 255,  # 归一化
    rotation_range=10,  # 旋转范围
    width_shift_range=0.1,  # 宽平移
    height_shift_range=0.1,  # 高平移
    shear_range=0.1,  # 剪切
    zoom_range=0.1,  # 缩放
    horizontal_flip=True,  # 随机将一半图像水平翻转
    fill_mode='nearest'  # 填充像素的方法
)

#验证集不进行数据增强
validation_datagen = ImageDataGenerator(
    rescale=1. / 255
)
#训练集加载
TRAINING_DIR = 'huotui/data'
training_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150, 150),
    class_mode='sparse'
)
#验证集加载
VALIDATION_DIR = 'huotui/test'
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150, 150),
    class_mode='sparse'
)

# ======== 模型构建 =41========
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    # 卷积层：输入参数：过滤器数量，过滤器尺寸，激活函数：relu， 输入图像尺寸
    tf.keras.layers.MaxPooling2D(2, 2),  # 池化层：增强特征
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),  # 输入参数：过滤器数量、过滤器尺寸、激活函数：relu
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    #tf.keras.layers.Dropout(0.1), # dropout层防止过拟合
    tf.keras.layers.Flatten(),  # 输入层
    tf.keras.layers.Dense(512, activation='relu'),  # 全连接隐层 神经元数量：512 ，激活函数：relu
    tf.keras.layers.Dense(3, activation='softmax')  # 输出用的是softmax 概率化函数 使得所有输出加起来为1 0-1之间
])

model.summary()

# ======== 模型参数编译 =========
model.compile(
    optimizer='rmsprop',#优化器选择
    loss='sparse_categorical_crossentropy',  # 损失函数： sparse_categorical_crossentropy int 类型的标签类的交叉熵
    metrics=['accuracy']
)

# ======== 模型训练 =========
# Note that this may take some time.
history = model.fit(
    training_generator,
    epochs=40,
    validation_data=validation_generator,
    verbose=1
)

model.save('htcmodel.h5')  # model 保存

# -----------------------------------------------------------
# Retrieve a list of list result on training and test data
# set for each training epoch
# -----------------------------------------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))  # Get number of epochs

# -----------------------------------------------------------
# Plot training and validation accuracy per epoch
# -----------------------------------------------------------
plt.plot(epochs, acc, 'r', label="tra_acc")
plt.plot(epochs, val_acc, 'b', label="val_acc")
plt.title("training and validation accuracy")
plt.legend(loc=0)
plt.grid(ls='--')  # 生成网格
plt.show()
# -----------------------------------------------------------
# Plot training and validation loss per epoch
# -----------------------------------------------------------
plt.plot(epochs, loss, 'r', label="train_loss")
plt.plot(epochs, val_loss, 'b', label="val_loss")
plt.title("training and validation loss")
plt.legend(loc=0)
plt.grid(ls='--')  # 生成网格
plt.show()
