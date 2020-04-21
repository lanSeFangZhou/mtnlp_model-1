import os
from ioflow.configure import read_configure
from ioflow.corpus import get_corpus_processor
from seq2annotation.input import generate_tagset, Lookuper, index_table_from_file
from seq2annotation.utils import create_dir_if_needed, create_file_dir_if_needed, create_or_rm_dir_if_needed
from tf_attention_layer.layers.global_attentioin_layer import GlobalAttentionLayer
from tf_crf_layer.layer import CRF
from tf_crf_layer.loss import ConditionalRandomFieldLoss
from tf_crf_layer.metrics import (
    crf_accuracy,
    SequenceCorrectness,
    sequence_span_accuracy,
)
from tokenizer_tools.tagset.converter.offset_to_biluo import offset_to_biluo

# tf.enable_eager_execution()
from typing import Union, Callable

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    BatchNormalization,
)

from deliverable_model.builtin.converter.identical_converters import ConverterForRequest, ConverterForResponse

import sys
sys.path.append('.')
sys.path.append('..')

from mtnlpmodel.trainer.utils import mt_export_as_deliverable_model, ConverterForMTResponse
from mtnlpmodel.trainer.lrset_util import SetLearningRate
from mtnlpmodel.mt_models import get_paragraph_vector



def main():

    # get configure
    config = read_configure()

    # get train/test corpus
    corpus = get_corpus_processor(config)
    corpus.prepare()
    train_data_generator_func = corpus.get_generator_func(corpus.TRAIN)
    eval_data_generator_func = corpus.get_generator_func(corpus.EVAL)

    corpus_meta_data = corpus.get_meta_info()

    # process str data to onehot
    ner_tags_data = generate_tagset(corpus_meta_data["tags"])
    cls_tags_data = corpus_meta_data["labels"]

    train_data = list(train_data_generator_func())
    eval_data = list(eval_data_generator_func())

    ner_tag_lookuper = Lookuper({v: i for i, v in enumerate(ner_tags_data)})
    cls_tag_lookuper = Lookuper({v: i for i, v in enumerate(cls_tags_data)})

    vocab_data_file = config.get("vocabulary_file")

    if not vocab_data_file:
        # load built in vocabulary file
        vocab_data_file = os.path.join(
            os.path.dirname(__file__), "../data/unicode_char_list.txt"
        )

    vocabulary_lookuper = index_table_from_file(vocab_data_file)

    def preprocss(data, maxlen, **kwargs):
        raw_x = []
        raw_y_ner = []
        raw_y_cls = []

        for offset_data in data:
            tags = offset_to_biluo(offset_data)
            label = offset_data.label
            words = offset_data.text

            tag_ids = [ner_tag_lookuper.lookup(i) for i in tags]
            label_id = cls_tag_lookuper.lookup(label)
            word_ids = [vocabulary_lookuper.lookup(i) for i in words]

            raw_x.append(word_ids)
            raw_y_ner.append(tag_ids)
            raw_y_cls.append(label_id)

        if maxlen is None:
            maxlen = max(len(s) for s in raw_x)

        print(">>> maxlen: {}".format(maxlen))

        x = tf.keras.preprocessing.sequence.pad_sequences(
            raw_x, maxlen, padding="post"
        )  # right padding

        y_ner = tf.keras.preprocessing.sequence.pad_sequences(
            raw_y_ner, maxlen, value=0, padding="post"
        )

        from keras.utils import to_categorical
        y_cls = np.array(raw_y_cls)
        y_cls = y_cls[:, np.newaxis]
        y_cls = to_categorical(y_cls, kwargs.get('cls_dims', 81))

        return x, y_ner, y_cls


    # get Parameters (controller)
    EPOCHS = config.get("epochs", 10)
    BATCHSIZE = config.get("batch_size", 32)
    LEARNINGRATE = config.get("learning_rate", 0.001)
    MAX_SENTENCE_LEN = config.get("max_sentence_len", 25)

    # get Parameters (model structure)
    EMBED_DIM = config.get("embedding_dim", 300)
    USE_ATTENTION_LAYER = config.get("use_attention_layer", False)
    BiLSTM_STACK_CONFIG = config.get("bilstm_stack_config", [])
    BATCH_NORMALIZATION_AFTER_EMBEDDING_CONFIG = config.get(
        "use_batch_normalization_after_embedding", False)
    BATCH_NORMALIZATION_AFTER_BILSTM_CONFIG = config.get(
        "use_batch_normalization_after_bilstm", False)
    CRF_PARAMS = config.get("crf_params", {})


    # get train/test data for training model
    vacab_size = vocabulary_lookuper.size()
    tag_size = ner_tag_lookuper.size()
    label_size = cls_tag_lookuper.size()

    train_x, train_y_ner, train_y_cls = preprocss(train_data, MAX_SENTENCE_LEN, **{'cls_dims':label_size})
    test_x, test_y_ner, test_y_cls = preprocss(eval_data, MAX_SENTENCE_LEN, **{'cls_dims':label_size})


    # build model
    input_length = MAX_SENTENCE_LEN
    input_layer = Input(shape=(input_length,), dtype='float', name='input_layer')

    # encoder
    with tf.keras.backend.name_scope("Encoder"):

        embedding_layer = Embedding(vacab_size,
                                    EMBED_DIM,
                                    mask_zero=True,
                                    input_length=input_length,
                                    name='embedding')(input_layer)

    # feature extractor
    with tf.keras.backend.name_scope("biLSTM"):
        if BATCH_NORMALIZATION_AFTER_EMBEDDING_CONFIG:
            embedding_layer = BatchNormalization()(embedding_layer)

        biLSTM = embedding_layer
        for bilstm_config in BiLSTM_STACK_CONFIG:
               biLSTM = Bidirectional(LSTM(return_sequences=True, **bilstm_config, name='biLSTM'))(biLSTM)

    if BATCH_NORMALIZATION_AFTER_BILSTM_CONFIG:
        biLSTM = BatchNormalization()(biLSTM)

    if USE_ATTENTION_LAYER:
        biLSTM = GlobalAttentionLayer()(biLSTM)

    # NER branch
    with tf.keras.backend.name_scope("NER_branch"):
        crf = CRF(tag_size, name="crf", **CRF_PARAMS)(biLSTM)
        loss_func = ConditionalRandomFieldLoss()


    # classification branch

    chosen = 'lstm_cls'
    with tf.keras.backend.name_scope("CLS_branch"):
        from tensorflow.keras.layers import Dense, Flatten, Dropout
        # add paragraph vector
        #paragraph_vector = get_paragraph_vector(embedding_layer)

        if chosen == "lstm_cls":
            cls_flat_lstm = Flatten()(biLSTM)
            #cls_flat_lstm = tf.keras.layers.concatenate([cls_flat_lstm, paragraph_vector])
            classification_dense = Dropout(0.2)(cls_flat_lstm)
            classification_dense = SetLearningRate(Dense(label_size, activation='sigmoid', name='CLS'), lr=0.001, is_ada=True)(classification_dense)

        elif chosen == "conv_cls":
            from tensorflow.keras.layers import Conv1D, MaxPooling1D
            embedding_layer = BatchNormalization()(embedding_layer)
            cls_conv_emb = Conv1D(32, 3, activation='relu', padding='same')(embedding_layer)
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
            classification_dense = Dense(label_size, activation='sigmoid', name='CLS')(classification_dense)



    # merge NER and Classification
    model = Model(inputs=[input_layer], outputs=[crf, classification_dense])


    model.summary()

    callbacks_list = []

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        #log_dir=create_dir_if_needed(config["summary_log_dir"])
        log_dir='.\\results\\summary_log_dir',
        batch_size=BATCHSIZE,
    )
    callbacks_list.append(tensorboard_callback)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(create_dir_if_needed(config["model_dir"]), "cp-{epoch:04d}.ckpt"),
        load_weights_on_restart=True,
        verbose=1,
    )
    callbacks_list.append(checkpoint_callback)

    metrics_list = []

    metrics_list.append(crf_accuracy)
    metrics_list.append(SequenceCorrectness())
    metrics_list.append(sequence_span_accuracy)

    # early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',  # early stop index
    #                                               patience=3,          # early stop delay epoch
    #                                               verbose=2,           # display mode
    #                                               mode='auto')
    # callbacks_list.append(early_stop)

    from mtnlpmodel.trainer.loss_func_util import FocalLoss
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNINGRATE, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam_optimizer,
                  #loss={'crf': loss_func, 'CLS': 'sparse_categorical_crossentropy'},
                  loss={'crf': loss_func, 'CLS': FocalLoss()},
                  loss_weights={'crf': 1., 'CLS': 100},  # set weight of loss
                  #metrics={'crf': SequenceCorrectness(), 'CLS': 'sparse_categorical_accuracy'} )
                  metrics={'crf': SequenceCorrectness(), 'CLS': 'categorical_accuracy'})

    model.fit(
        train_x,
        {'crf': train_y_ner, 'CLS': train_y_cls},
        epochs=EPOCHS,
        batch_size=BATCHSIZE,
        validation_data=[test_x,  {'crf': test_y_ner, 'CLS': test_y_cls}],
        callbacks=callbacks_list,
    )


    model.save(create_file_dir_if_needed(config["h5_model_file"]))
    model.save_weights(create_file_dir_if_needed(config["h5_weights_file"]))

    tf.keras.experimental.export_saved_model(
        model, create_or_rm_dir_if_needed(config["saved_model_dir"])
    )


    mt_export_as_deliverable_model(
        create_dir_if_needed(config["deliverable_model_dir"]),
        keras_saved_model=config["saved_model_dir"],
        converter_for_request=ConverterForRequest(),
        converter_for_response=ConverterForMTResponse(),
        lookup_tables={'vocab_lookup':vocabulary_lookuper,
                       'tag_lookup':ner_tag_lookuper,
                       'label_lookup':cls_tag_lookuper},
        padding_parameter={"maxlen": MAX_SENTENCE_LEN, "value": 0, "padding": "post"},
        addition_model_dependency=["tf-crf-layer"],
        custom_object_dependency=["tf_crf_layer"],
    )





if __name__ == "__main__":
    main()
