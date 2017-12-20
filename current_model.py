
def InputBlock(x):
    #conv layers for input
    x = BatchNormalization()(x)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x) 
    return(x)

main_input = Input(shape=(75,75,2), name='main_input')
aux_input = Input(shape=(75,75,3), name='aux_input')
aux_input_wavelet = Input(shape=(75,75,4), name='aux_input_wavelet')
aux_input_nn = Input(shape=(75,75,4), name='aux_input_nn')
aux_input_filter = Input(shape=(75,75,4), name='aux_input_filter')
aux_input_dilated = Input(shape=(75,75,4), name='aux_input_dilated')

x1 = InputBlock(main_input)
x2 = InputBlock(aux_input)
x3 = InputBlock(aux_input_wavelet)
x4 = model_denoise(aux_input_nn)
x4 = InputBlock(x4)
x5 = InputBlock(aux_input_filter)
x6 = InputBlock(aux_input_dilated)

x = Concatenate(axis=3)([x1,x2,x3,x4,x5,x6])
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

#conv-block
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

#conv-block
x = Conv2D(256, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
    
#flatten
x = Flatten()(x)
angle_input = Input(shape=[1], name='angle_input')
x1 = BatchNormalization()(angle_input)
merged = Concatenate()([x, x1])

#dense-block
x = Dense(513, activation='relu')(merged)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

#dense-block
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

main_output = Dense(1, activation='sigmoid', name='main_output')(x)
model_f = Model(inputs=[main_input,aux_input, 
                        aux_input_wavelet, aux_input_nn, aux_input_filter, aux_input_dilated,
                        angle_input,], 
                        outputs=[main_output])

model_f.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0),
loss='binary_crossentropy',
metrics=['accuracy'])