import tensorflow as tf

class SetLearningRate:

    """
        层的一个包装，用来设置当前层的学习率

        转载自：https://kexue.fm/archives/6418
    """

    def __init__(self, layer, lr=0.01, is_ada=False):
        self.layer = layer
        self.lamb = lr # 学习率比例
        self.is_ada = is_ada # 是否自适应学习率优化器

    def __call__(self, inputs):
        with tf.keras.backend.name_scope(self.layer.name):
            if not self.layer.built:
                input_shape = tf.keras.backend.int_shape(inputs)
                self.layer.build(input_shape)
                self.layer.built = True
                if self.layer._initial_weights is not None:
                    self.layer.set_weights(self.layer._initial_weights)
        for key in ['kernel', 'bias', 'embeddings', 'depthwise_kernel', 'pointwise_kernel', 'recurrent_kernel', 'gamma', 'beta']:
            if hasattr(self.layer, key):
                weight = getattr(self.layer, key)
                if self.is_ada:
                    lamb = self.lamb # 自适应学习率优化器直接保持lamb比例
                else:
                    lamb = self.lamb**0.5 # SGD（包括动量加速），lamb要开平方
                tf.keras.backend.set_value(weight, tf.keras.backend.eval(weight) / lamb) # 更改初始化
                # setattr(self.layer, key, weight*lamb)
                tf.keras.backend.set_value(weight, tf.keras.backend.eval(weight) * lamb) # 修正

        return self.layer(inputs)


