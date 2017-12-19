
main_input = Input(shape=(75,75,2), name='main_input')
aux_input = Input(shape=(75,75,3), name='aux_input')

#conv layers for main_input
x1 = BatchNormalization()(main_input)
x1 = Conv2D(64, (3,3), activation='relu')(x1)
x1 = MaxPooling2D((2, 2), strides=(2, 2))(x1)
x1 = BatchNormalization()(x1)
x1 = Dropout(0.2)(x1)

#conv layers for aux_input
x2 = BatchNormalization()(aux_input)
x2 = Conv2D(64, (3,3), activation='relu')(x2)
x2 = MaxPooling2D((2, 2), strides=(2, 2))(x2)
x2 = BatchNormalization()(x2)
x2 = Dropout(0.2)(x2)

x = Concatenate(axis=3)([x1,x2])
    
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
merged = Concatenate()([x, angle_input])

    #dense-block
x = Dense(513, activation='relu')(merged)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

    #dense-block
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

main_output = Dense(1, activation='sigmoid', name='main_output')(x)
model_f = Model(inputs=[main_input,aux_input,angle_input], outputs=[main_output])

model_f.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0),
loss='binary_crossentropy',
metrics=['accuracy'])