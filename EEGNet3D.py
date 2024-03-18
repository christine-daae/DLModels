from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, SpatialDropout2D
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import AveragePooling3D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

from DepthwiseConv3D import DepthwiseConv3D

AVERAGE_POOLING_SIZE1 = (1, 1, 4)

AVERAGE_POOLING_SIZE2 = (1, 1, 8)


def EEGNet3D():
	input1 = Input(shape = (5, 5, 512, 1))
	block1 = Conv3D(16, (3, 3, 5), (2, 2, 4), padding='same', activation='linear', use_bias=False)(input1)
	block1 = BatchNormalization()(block1)
	block1 = Activation('elu')(block1)
	print(block1.shape)
	branch1_1 = Conv3D(16, (2, 2, 3), padding='same', activation='linear', use_bias=False)(block1)
	branch1_1 = BatchNormalization()(branch1_1)
	branch1_1 = Activation('elu')(branch1_1)
	branch1_1 = Dropout(0.5)(branch1_1)
	branch1_1 = Conv3D(16, (2, 2, 3), padding='same', activation='linear', use_bias=False)(branch1_1)
	branch1_1 = BatchNormalization()(branch1_1)
	branch1_1 = Activation('elu')(branch1_1)
	branch1_1 = Dropout(0.5)(branch1_1)
	branch1_2 = Conv3D(16, (3, 3, 5), padding='same', activation='linear', use_bias=False)(block1)
	branch1_2 = BatchNormalization()(branch1_2)
	branch1_2 = Activation('elu')(branch1_2)
	branch1_2 = Dropout(0.5)(branch1_2)
	branch1_2 = Conv3D(16, (3, 3, 5), padding='same', activation='linear', use_bias=False)(branch1_2)
	branch1_2 = BatchNormalization()(branch1_2)
	branch1_2 = Activation('elu')(branch1_2)
	branch1_2 = Dropout(0.5)(branch1_2)
	block1 = concatenate([branch1_1, branch1_2], axis=-1)

	block2 = Conv3D(16, (2, 2, 3), (2, 2, 2), padding='same', activation='linear', use_bias=False)(block1)
	block2 = BatchNormalization()(block2)
	block2 = Activation('elu')(block2)
	print("block2", block2.shape)
	branch2_1 = Conv3D(16, (2, 2, 3), padding='same', activation='linear', use_bias=False)(block2)
	branch2_1 = BatchNormalization()(branch2_1)
	branch2_1 = Activation('elu')(branch2_1)
	branch2_1 = Dropout(0.5)(branch2_1)
	branch2_1 = Conv3D(16, (2, 2, 3), padding='same', activation='linear', use_bias=False)(branch2_1)
	branch2_1 = BatchNormalization()(branch2_1)
	branch2_1 = Activation('elu')(branch2_1)
	branch2_1 = Dropout(0.5)(branch2_1)
	branch2_2 = Conv3D(16, (3, 3, 5), padding='same', activation='linear', use_bias=False)(block2)
	branch2_2 = BatchNormalization()(branch2_2)
	branch2_2 = Activation('elu')(branch2_2)
	branch2_2 = Dropout(0.5)(branch2_2)
	branch2_2 = Conv3D(16, (3, 3, 5), padding='same', activation='linear', use_bias=False)(branch2_2)
	branch2_2 = BatchNormalization()(branch2_2)
	branch2_2 = Activation('elu')(branch2_2)
	branch2_2 = Dropout(0.5)(branch2_2)
	block2 = concatenate([branch2_1, branch2_2], axis=-1)

	block3 = Conv3D(16, (2, 2, 3), (2, 2, 2), padding='same', activation='linear', use_bias=False)(block2)
	block3 = BatchNormalization()(block3)
	block3 = Activation('elu')(block3)
	print("block3", block3.shape)
	branch3_1 = Conv3D(16, (2, 2, 3), padding='same', activation='linear', use_bias=False)(block3)
	branch3_1 = BatchNormalization()(branch3_1)
	branch3_1 = Activation('elu')(branch3_1)
	branch3_1 = Dropout(0.5)(branch3_1)
	branch3_1 = Conv3D(16, (2, 2, 3), padding='same', activation='linear', use_bias=False)(branch3_1)
	branch3_1 = BatchNormalization()(branch3_1)
	branch3_1 = Activation('elu')(branch3_1)
	branch3_1 = Dropout(0.5)(branch3_1)
	branch3_2 = Conv3D(16, (3, 3, 5), padding='same', activation='linear', use_bias=False)(block3)
	branch3_2 = BatchNormalization()(branch3_2)
	branch3_2 = Activation('elu')(branch3_2)
	branch3_2 = Dropout(0.5)(branch3_2)
	branch3_2 = Conv3D(16, (3, 3, 5), padding='same', activation='linear', use_bias=False)(branch3_2)
	branch3_2 = BatchNormalization()(branch3_2)
	branch3_2 = Activation('elu')(branch3_2)
	branch3_2 = Dropout(0.5)(branch3_2)
	block3 = concatenate([branch3_1, branch3_2], axis=-1)


	flatten = Flatten(name = 'flatten')(block3)
	dense1 = Dense(32)(flatten)
	dense1 = BatchNormalization()(dense1)
	dense1 = Activation('elu')(dense1)
	dense1 = Dropout(0.5)(dense1)

	dense1 = Dense(32)(dense1)
	dense1 = BatchNormalization()(dense1)
	dense1 = Activation('elu')(dense1)
	dense1 = Dropout(0.5)(dense1)

	out = Dense(2, activation='softmax')(dense1)
	return Model(inputs=input1, outputs=out)


