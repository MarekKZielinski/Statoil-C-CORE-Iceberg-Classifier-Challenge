
def SimpleInceptionBlock(x, filters=64, dropout=0.2, prefix=''):
    #path 1
    x1 = Conv2D(filters, (1,1), padding='same', activation='relu', name=prefix + '11_Conv_1_1')(x)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name=prefix + '12_MaxPool_2_2')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(dropout)(x1) 
    #path 2
    x2 = Conv2D(filters, (3,3), activation='relu', padding='same', name=prefix + '21_Conv_3_3')(x)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name=prefix + '22_MaxPool_2_2')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(dropout)(x2) 
    #path 3
    x3 = Conv2D(filters, (1,1), activation='relu', padding='same', name=prefix + '31_Conv_1_1')(x)
    x3 = Conv2D(filters, (5,5), activation='relu', padding='same', name=prefix + '32_Conv_5_5')(x3)
    x3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name=prefix + '33_MaxPool_2_2')(x3)
    x3 = BatchNormalization()(x3)
    x3 = Dropout(dropout)(x3)
    #concatenate
    x = Concatenate(axis=3, name=prefix+'_Inception_end')([x1,x2,x3])
    return x

def ConvBlock(x, filters=64, dropout=0.2, prefix=''):
    x = Conv2D(filters, (3,3), activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x) 
    
def InputBlock(x, dropout=0.2, prefix=''):
    #conv layers for input
    x = BatchNormalization()(x)
    x = SimpleInceptionBlock(x, prefix=prefix+'_1_')
    x = SimpleInceptionBlock(x, prefix=prefix+'_2_')
    #x = ConvBlock(x)
    #x = ConvBlock(x)
    return(x)

main_input = Input(shape=(75,75,2), name='main_input')
aux_input = Input(shape=(75,75,3), name='aux_input')
    #aux_input_nn = Input(shape=(75,75,4), name='aux_input_nn')

x1 = InputBlock(main_input, prefix='m_input')
x2 = InputBlock(aux_input, prefix='a_input')
    #x3 = model_denoise(aux_input_nn)
    #x3 = InputBlock(x3,dropout=0.3, prefix='a_input_nn')

    #x = x1
x = Concatenate(axis=3)([x1,x2])
    #x = BatchNormalization()(x)
    #x = Dropout(0.2)(x)

    #conv-blocks
#x = ConvBlock(x, filters=128)
#x = ConvBlock(x, filters=256)

    #inception blocks
x = SimpleInceptionBlock(x, filters=128, prefix='main_1')
x = SimpleInceptionBlock(x, filters=256, prefix='main_2')

    #flatten
x = Flatten()(x)
angle_input = Input(shape=[1], name='angle_input')
    #x1 = BatchNormalization()(angle_input)
merged = Concatenate()([x, angle_input])
    
    #dense-block
x = Dense(1025, activation='relu')(merged)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
    
    #dense-block
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
    
main_output = Dense(1, activation='sigmoid', name='main_output')(x)
model_f = Model(inputs=[main_input,aux_input, 
                            #aux_input_nn, 
                            angle_input], 
                            outputs=[main_output])

model_f.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0),
    loss='binary_crossentropy',
    metrics=['accuracy'])
    