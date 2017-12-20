
def InceptionBlock(x, n_kernels=32, prefix=''):
    path1 = Conv2D(n_kernels, (1,1), padding='same', activation='relu', name=prefix + '11_Conv_1_1')(x)
    
    path2 = Conv2D(n_kernels, (1,1), padding='same', activation='relu', name=prefix + '21_Conv_1_1')(x)
    path2 = Conv2D(n_kernels, (3,3), padding='same', activation='relu', name=prefix + '22_Conv_3_3')(path2)
    
    path3 = Conv2D(n_kernels, (1,1), padding='same', activation='relu', name=prefix + '31_Conv_1_1')(x)
    path3 = Conv2D(n_kernels, (5,5), padding='same', activation='relu', name=prefix + '32_Conv_5_5')(path3)
    
    path4 = MaxPooling2D((3,3), strides=(1,1), padding='same', name=prefix + '41_MaxPool_3_3')(x)
    path4 = Conv2D(n_kernels, (1,1), padding='same', activation='relu', name=prefix + '42_Conv_1_1')(path4)
    
    out = Concatenate(axis=3, name=prefix+'_Inception_end')([path1,path2,path3,path4])
    return(out)

def InputBlock(x, dropout=0.2, prefix=''):
    #conv layers for input
    x = BatchNormalization()(x)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x) 
    x = InceptionBlock(x,prefix=prefix)
    return(x)

main_input = Input(shape=(75,75,2), name='main_input')
aux_input = Input(shape=(75,75,3), name='aux_input')
aux_input_nn = Input(shape=(75,75,4), name='aux_input_nn')

x1 = InputBlock(main_input, prefix='m_input')
x2 = InputBlock(aux_input, prefix='a_input')
x3 = model_denoise(aux_input_nn)
x3 = InputBlock(x3,dropout=0.2, prefix='a_input_nn')

x = Concatenate(axis=3)([x1,x2,x3])
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

x = InceptionBlock(x, prefix='main_path1')
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = InceptionBlock(x, prefix='main_path2')
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
                        aux_input_nn, 
                        angle_input,], 
                        outputs=[main_output])

model_f.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0),
loss='binary_crossentropy',
metrics=['accuracy'])