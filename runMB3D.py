from EEGNet3D import EEGNet3D
from LoadData3 import Get_data

import os
# 选择编号为0的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import numpy as np

# 设置随机数种子以确保结果可复现
np.random.seed(42)

# # 生成500个样本
# num_samples = 500
# data = np.random.randn(num_samples, 9, 9, 400, 1)
#
# # 生成对应的标签，这里假设是二分类，标签为0或1
# labels = np.random.randint(2, size=(num_samples,))

data, labels = Get_data(13)

class CustomCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		print("Finished epoch {} with Loss: {} and Accuracy: {}".format(epoch, logs['loss'], logs['accuracy']))

tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)


loss_functions = ('sparse_categorical_crossentropy', 'mean_absolute_error')

init_lr = 0.001

opt = tf.keras.optimizers.Adam(init_lr)



seed = 4

print("Starting K-Fold")
i = 1
cvscores = []
kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
for train, test in kfold.split(data, labels):
	print("Performing K-Fold", i)
	model = EEGNet3D()
	model.compile(loss=loss_functions[0], optimizer=opt, metrics=['accuracy'])
	model.fit(data[train], labels[train], epochs=30, verbose=0, callbacks=[CustomCallback()])
	scores = model.evaluate(data[test], labels[test], verbose=0, callbacks=[CustomCallback()])
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1]*100)
	i += 1

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
acc = np.mean(cvscores)
std = np.std(cvscores)

print("Accuracy: %.2f" % (acc))

#print("Testing alternate Subject #2")

#_, acc = model.evaluate(X_2_val, Y_2_val, verbose=0, callbacks=[CustomCallback()])

#print("Accuracy: %.2f" % (acc*100))

