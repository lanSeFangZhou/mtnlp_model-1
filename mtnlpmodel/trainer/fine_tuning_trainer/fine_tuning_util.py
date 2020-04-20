import tensorflow as tf
from tf_attention_layer.layers.global_attentioin_layer import GlobalAttentionLayer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    BatchNormalization,
    Dense,
    Input,
    Flatten
)

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

import sys

sys.path.append('.')
sys.path.append('../../../')

from mtnlpmodel.trainer.utils import mt_export_as_deliverable_model

from mtnlpmodel.trainer.lrset_util import SetLearningRate




def backbone_network(bilstm_configs,
                     input_layer=Input,
                     vacab_size=7540,
                     EMBED_DIM=300,
                     input_length=45,):

    # Encoder
    with tf.keras.backend.name_scope("Encoder"):
        embedding_layer = Embedding(vacab_size,
                                    EMBED_DIM,
                                    mask_zero=True,
                                    input_length=input_length,
                                    name='embedding')(input_layer)

    # Extractor
    with tf.keras.backend.name_scope("biLSTM"):
        embedding_layer = BatchNormalization()(embedding_layer)
        biLSTM = embedding_layer
        for bilstm_config in bilstm_configs:
            biLSTM = Bidirectional(LSTM(return_sequences=True, **bilstm_config, name='biLSTM'))(biLSTM)
        biLSTM = BatchNormalization()(biLSTM)
        biLSTM = GlobalAttentionLayer()(biLSTM)

    backbone = biLSTM

    return backbone



def add_new_layer(raw_model, layer):

    return layer(raw_model)



def transfer_learning(input_shape,
                      input_layer,
                      base_model,
                      new_task_output_layer,
                      index_dict_for_approaching,
                      backbone_model_path,
                      warm_start_list,):

    cls_layer = new_task_output_layer
    base_model = Flatten()(base_model)
    output_layer = add_new_layer(base_model, cls_layer)
    new_model = Model(input_layer, output_layer, name='new_model')
    new_model.compile(optimizer=index_dict_for_approaching['optimizer'],
                      loss=index_dict_for_approaching['loss'],
                      metrics=index_dict_for_approaching['metrics'])

    freeze_layers = [layer for layer in new_model.layers if layer.name in warm_start_list]
    trainable_layers = [layer for layer in new_model.layers if layer.name not in warm_start_list]

    for layer in freeze_layers:
        layer.trainable = False

    for layer in trainable_layers:
        layer.trainable = True

    new_model.load_weights(backbone_model_path, by_name=True)

    return new_model





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

    def preprocss(data, maxlen):
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

        y_cls = np.array(raw_y_cls)
        y_cls = y_cls[:, np.newaxis]

        return x, y_ner, y_cls


    # get Parameters (controller)
    EPOCHS = config.get("epochs", 10)
    BATCHSIZE = config.get("batch_size", 32)
    LEARNINGRATE = config.get("learning_rate", 0.0001)
    MAX_SENTENCE_LEN = config.get("max_sentence_len", 25)

    # get Parameters (model structure)
    EMBED_DIM = config.get("embedding_dim", 300)
    BiLSTM_STACK_CONFIG = config.get("bilstm_stack_config", [])


    # get train/test data for training model
    train_x, train_y_ner, train_y_cls = preprocss(train_data, MAX_SENTENCE_LEN)
    test_x, test_y_ner, test_y_cls = preprocss(eval_data, MAX_SENTENCE_LEN)

    vacab_size = vocabulary_lookuper.size()
    # tag_size = ner_tag_lookuper.size()
    label_size = cls_tag_lookuper.size()




# finetuning correlation code

    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNINGRATE, beta_1=0.9, beta_2=0.999, amsgrad=False)
    index_dict= {'optimizer': adam_optimizer,
                 'loss': 'sparse_categorical_crossentropy',
                 'metrics': ['sparse_categorical_accuracy']
                }

    warm_start_list = ['embedding', 'bidirectional', 'batch_normalization']  # layer in list is frozen

    backbone_model_path = './mtnlpmodel/trainer/fine_tuning_trainer/save_weights/weights.h5'

    output_dims = label_size



# model structure correlation code

    # define new_layer for the task
    new_task_output_layer = Dense(output_dims, activation='softmax')  # new softmax layer -> output

    input_shape =MAX_SENTENCE_LEN
    input_layer = Input(shape=(input_shape,), dtype='int32', name='input_layer')   # input

    # backbone + transfer_learning function can use backbone to do some different job
    # what you need is to define a new layer which is new_task_output_layer below
    # this code is a sample for only text classification
    # backbone output is biLSTM's output, so transfer_learning add a flatten layer to connect dense layer
    # you can modify the structure in function transfer_learning to match your task demand
    base_model = backbone_network(BiLSTM_STACK_CONFIG,
                                  input_layer=input_layer,
                                  vacab_size=vacab_size,
                                  EMBED_DIM=EMBED_DIM,
                                  input_length=MAX_SENTENCE_LEN,
                                  )


    new_model = transfer_learning(input_shape,
                                  input_layer,
                                  base_model,
                                  new_task_output_layer,
                                  index_dict,
                                  backbone_model_path,
                                  warm_start_list)



# model output info correlation code
    new_model.summary()

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

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',  # early stop index
                                                  patience=3,          # early stop delay epoch
                                                  verbose=2,           # display mode
                                                  mode='auto')
    callbacks_list.append(early_stop)

    new_model.fit(
        train_x,
        train_y_cls,
        epochs=EPOCHS,
        batch_size=BATCHSIZE,
        validation_data=[test_x, test_y_cls],
        callbacks=callbacks_list,
    )


    new_model.save(create_file_dir_if_needed(config["h5_model_file"]))

    tf.keras.experimental.export_saved_model(
        new_model, create_or_rm_dir_if_needed(config["saved_model_dir"])
    )

    mt_export_as_deliverable_model(
        create_dir_if_needed(config["deliverable_model_dir"]),
        keras_saved_model=config["saved_model_dir"],
        vocabulary_lookup_table=vocabulary_lookuper,
        tag_lookup_table=ner_tag_lookuper,
        label_lookup_table=cls_tag_lookuper,
        padding_parameter={"maxlen": MAX_SENTENCE_LEN, "value": 0, "padding": "post"},
        addition_model_dependency=["tf-crf-layer"],
        custom_object_dependency=["tf_crf_layer"],
    )




if __name__ == '__main__':

    main()




