# some models for classification

def lstm_cls(input_layer, output_dim):
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    cls_flat = Flatten()(input_layer)
    cls_dropout = Dropout(0.2)(cls_flat)
    output_layer = Dense(output_dim, activation='sigmoid', name='cls_Dense')(cls_dropout)

    return output_layer



def dilated_cnn_cls(input_layer, output_dim):
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dense, Dropout, Flatten
    input_layer = BatchNormalization()(input_layer)
    cls_conv_emb = Conv1D(32, 3, activation='relu', padding='same')(input_layer)
    cls_conv_emb = Conv1D(64, 3, activation='relu', padding='same')(cls_conv_emb)
    cls_conv_emb = MaxPooling1D(2)(cls_conv_emb)
    cls_conv_emb = Conv1D(128, 3, activation='relu', dilation_rate=1, padding='same')(cls_conv_emb)
    cls_conv_emb = Conv1D(128, 3, activation='relu', dilation_rate=2, padding='same')(cls_conv_emb)
    cls_conv_emb = Conv1D(128, 3, activation='relu', dilation_rate=5, padding='same')(cls_conv_emb)
    cls_conv_emb = Conv1D(256, 1, activation='relu', padding='same')(cls_conv_emb)
    cls_conv_emb = MaxPooling1D(2)(cls_conv_emb)

    cls_flat = BatchNormalization()(cls_conv_emb)
    cls_flat = Flatten()(cls_flat)
    classification_dense = Dropout(0.2)(cls_flat)
    output_layer = Dense(output_dim, activation='sigmoid', name='CLS')(classification_dense)

    return output_layer



def textcnn_cls(input_layer, output_dim):
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, concatenate, BatchNormalization, Dense, Dropout, Flatten
    kernel_sizes = [2, 3, 4]
    pooling_out = []
    cls_conv_layer = input_layer
    for kernel_size in kernel_sizes:
        cls_conv_layer = Conv1D(filters=128, kernel_size=kernel_size, strides=1)(cls_conv_layer)
        cls_pooling_layer = MaxPooling1D(pool_size=int(cls_conv_layer.shape[1]))(cls_conv_layer)
        pooling_out.append(cls_pooling_layer)
        #print("kernel_size: %s \t c.shape: %s \t p.shape: %s" % (kernel_size, str(cls_conv_layer.shape), str(cls_pooling_layer.shape)))
    pool_output = concatenate([p for p in pooling_out])
    cls_flat = Flatten()(pool_output)
    cls_flat = Dropout(0.2)(cls_flat)
    cls_dense = Dense(output_dim, activation='softmax')(cls_flat)

    return cls_dense



def fasttext_cls(input_layer, output_dim):
    from tensorflow.keras.layers import Dense
    # fast_text generally use embedding layer as its input_layer
    # directly connect dense to classify
    return Dense(output_dim, activation='softmax')(input_layer)



def get_paragraph_vector(input_layer):
    from tensorflow.keras.layers import Conv1D, Flatten ,MaxPooling1D
    embedding_layer = input_layer
    paragraph_vector = Conv1D(32, 3, activation='relu', padding='same')(embedding_layer)
    paragraph_vector = Conv1D(64, 3, activation='relu', padding='same')(paragraph_vector)
    paragraph_vector = MaxPooling1D(2)(paragraph_vector)
    paragraph_vector = Conv1D(128, 3, activation='relu', dilation_rate=1, padding='same')(paragraph_vector)
    paragraph_vector = Conv1D(128, 3, activation='relu', dilation_rate=2, padding='same')(paragraph_vector)
    paragraph_vector = Conv1D(128, 3, activation='relu', dilation_rate=5, padding='same')(paragraph_vector)
    paragraph_vector = Conv1D(256, 1, activation='relu', padding='same')(paragraph_vector)
    paragraph_vector = MaxPooling1D(2)(paragraph_vector)
    paragraph_vector = Flatten()(paragraph_vector)

    return paragraph_vector