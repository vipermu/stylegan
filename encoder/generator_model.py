import tensorflow as tf
import numpy as np
import dnnlib.tflib as tflib
from functools import partial


def create_stub(name, batch_size):
    return tf.constant(0, dtype='float32', shape=(batch_size, 0))

def create_variable_for_generator(name, batch_size, mode):
    with tf.variable_scope("dlatents", reuse=tf.AUTO_REUSE):
        if mode == 'dlatents':
            
            learnable_dlatents = tf.get_variable('learnable_dlatents',
                                shape=(batch_size, 1, 512),
                                dtype='float32',
                                initializer=tf.initializers.random_normal())
            return tf.tile(learnable_dlatents, (1,18,1))
        else:
            return tf.get_default_graph().get_tensor_by_name("G_mapping_1/_Run/concat:0") #tf.convert_to_tensor(np.load('./outputs/temp_latents_trump.npy')[0]) #

def create_variable_for_mapping(name, batch_size):
    return tf.get_variable('learnable_latents',
                           shape=(batch_size, 512),
                           dtype='float32',
                           initializer=tf.initializers.zeros())

""" 
    This class creates Variable Tensors for the latents/dlatents so we can train them.
"""

class Generator:
    def __init__(self, model, randomize_noise=False, mode = 'dlatents'):
        batch_size = 1
        self.model = model
        
        if mode == 'dlatents':
            self.initial_values = np.zeros((batch_size, 18, 512)) # This value is just to make the next function run. The dlatents value is initialized later
            self.model.components.synthesis.run(self.initial_values,
                                        randomize_noise=randomize_noise, minibatch_size=batch_size,
                                        custom_inputs=[partial(create_variable_for_generator, batch_size=batch_size,
                                                        mode=mode),
                                                        partial(create_stub, batch_size=batch_size)],
                                        structure='fixed')

        # Case where we want to optimize latent vectors
        else:
            self.initial_values = np.zeros((1, 512))
            self.model.components.mapping.run(self.initial_values, None,
                                        minibatch_size=batch_size,
                                        custom_inputs=[partial(create_variable_for_mapping, batch_size=batch_size),
                                                        partial(create_stub, batch_size=batch_size)],
                                        structure='fixed')

            self.initial_dlatents = np.random.rand(batch_size, 18, 512)*0.01
            self.model.components.synthesis.run(self.initial_dlatents,
                                        randomize_noise=randomize_noise, minibatch_size=batch_size,
                                        custom_inputs=[partial(create_variable_for_generator, batch_size=batch_size,
                                                        mode=mode),
                                                        partial(create_stub, batch_size=batch_size)],
                                        structure='fixed')

        self.sess = tf.get_default_session()
        self.graph = tf.get_default_graph()

        train_writer = tf.summary.FileWriter('./tensorboard_logs/generator', self.sess.graph) # Visualization of the new graph with the added variables
         
        self.latent_variable = next(v for v in tf.global_variables() if 'learnable_' in v.name)

        if mode == 'dlatents':
            self.set_latents(np.zeros((batch_size, 1, 512)))
        else:
            self.set_latents(self.initial_values)
        
        self.generator_output = self.graph.get_tensor_by_name('G_synthesis_1/_Run/concat:0') # Found in the tensorboard graph
        self.generated_image = tflib.convert_images_to_uint8(self.generator_output, nchw_to_nhwc=True, uint8_cast=False)
        self.generated_image_uint8 = tf.saturate_cast(self.generated_image, tf.uint8) # Converts the image to the dtype uint8

    def set_latents(self, latents):
        self.sess.run(tf.assign(self.latent_variable, latents))

    def get_latents(self):
        return self.sess.run(self.latent_variable)

    def generate_images(self):
        return self.sess.run(self.generated_image_uint8)
