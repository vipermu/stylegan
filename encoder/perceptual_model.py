import numpy as np
import os

import tensorflow as tf
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import keras.backend as K

import dnnlib.tflib as tflib

import PIL

def load_image(img_path, img_size):
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img = np.expand_dims(img, 0)
    preprocessed_images = preprocess_input(img)
    return preprocessed_images


class PerceptualModel:
    """PerceptialModel
    
    Returns:
        Class Object -- Creates a class with the required functions to train a tensor w.r.t. a perceptual loss

    Functions:
        build_perceptual_model -- defines a pretrained vgg16 model and compute the differences between the 
        generated image and the reference one in multiple layers
    """

    def __init__(self, img_size, layer=9, sess=None):
        self.sess = tf.get_default_session() if sess is None else sess
        K.set_session(self.sess)
        self.img_size = img_size
        self.layer = layer

        self.perceptual_model = None
        self.ref_img_features = None
        self.features_weight = None
        self.loss = None

        self.perceptual_models_list = []
        self.ref_img_features_list = []

    def build_perceptual_model(self, generated_image_tensor, mode, ref_dlatents_dir = '', ref_images=''):
        
        vgg16 = VGG16(include_top=False, input_shape=(self.img_size, self.img_size, 3))
        # print(vgg16.summary())
        
        self.loss_1 = self.get_loss_from_model(vgg16, self.layer, generated_image_tensor)
        self.set_reference_images(ref_images)
        self.loss_2 = self.get_loss_from_model(vgg16, self.layer-4, generated_image_tensor)
        self.set_reference_images(ref_images)
        self.loss_3 = self.get_loss_from_model(vgg16, self.layer+4, generated_image_tensor)
        self.set_reference_images(ref_images)
        self.loss_4 = self.get_loss_from_model(vgg16, self.layer-+8, generated_image_tensor)
        self.set_reference_images(ref_images)

        # self.loss = (0.4*self.loss_1 + 0.4*self.loss_2 + 0.2*self.loss_3*20 + 0.2*self.loss_3*50)
        self.loss = (1e-4/12*self.loss_1 + 1e-5*self.loss_2 + 4e-4*self.loss_3 + 1e-2/6*self.loss_4)
      

        if mode == 'latents':
            self.ref_dlatents = tf.convert_to_tensor(np.load(ref_dlatents_dir)[0])
            # self.ref_dlatents = tf.convert_to_tensor(np.zeros((1,18,512)))
            self.dlatents = tf.get_default_graph().get_tensor_by_name('G_mapping_1/_Run/G_mapping/Dense7/LeakyReLU/IdentityN:0')
            print(self.sess.run(self.ref_dlatents)[0])
            print(self.sess.run(self.dlatents)[0])

            self.loss_4 = tf.losses.mean_squared_error(self.ref_dlatents[0], self.dlatents)
            # self.loss = self.loss_4*10
            # self.loss = (0.4*self.loss_1 + 0.05*self.loss_2 + 0.4*self.loss_3*25 + 0.4*self.loss_4*100)

    
    def get_loss_from_model(self, model, layer, generated_image_tensor):
        self.perceptual_model = Model(model.input, model.layers[layer].output)


        generated_image = preprocess_input(tf.image.resize_images(generated_image_tensor,
                                                                (self.img_size, self.img_size), method=1))
        generated_img_features = self.perceptual_model(generated_image)

        with tf.variable_scope("refFeatures", reuse=tf.AUTO_REUSE):
            self.ref_img_features = tf.get_variable('ref_img_features_' + str(layer), shape=generated_img_features.shape,
                                                    dtype='float32', initializer=tf.initializers.zeros())

        self.perceptual_models_list.append(self.perceptual_model)
        self.ref_img_features_list.append(self.ref_img_features)
        
        self.sess.run([self.ref_img_features.initializer])

        return tf.losses.mean_squared_error(self.ref_img_features,
                                                generated_img_features)


    
    def set_reference_images(self, img_path):
        loaded_image = load_image(img_path, self.img_size)
        for idx, model in enumerate(self.perceptual_models_list):
            image_features = model.predict_on_batch(loaded_image)
            self.sess.run(tf.assign(self.ref_img_features_list[idx], image_features))

    def generate_images(self, out_dir, img_name, i):
        generator_output = tf.get_default_graph().get_tensor_by_name('G_synthesis_1/_Run/concat:0')
        generated_image = tflib.convert_images_to_uint8(generator_output, nchw_to_nhwc=True, uint8_cast=False)
        generated_image_uint8 = tf.saturate_cast(generated_image, tf.uint8) # Converts the image to the dtype uint8
        generated_images = self.sess.run(generated_image_uint8)
         # Generate images from found dlatents and save them
        img_array = generated_images[0]
        img = PIL.Image.fromarray(img_array, 'RGB')
        img.save(os.path.join(out_dir, str(img_name) + '_' + str(i) + '.png'), 'PNG')


    def optimize(self, vars_to_optimize, iterations=500, learning_rate=0.009, **kwargs):        
        vars_to_optimize = vars_to_optimize if isinstance(vars_to_optimize, list) else [vars_to_optimize]
        optimizer = tf.train.AdamOptimizer(learning_rate)
        min_op = optimizer.minimize(self.loss, var_list=[vars_to_optimize])
        self.sess.run(tf.variables_initializer(optimizer.variables()))

        train_writer = tf.summary.FileWriter('./tensorboard_logs/perceptual', self.sess.graph)

        for i in range(iterations):
            if i % 50 == 0:
                np.save(os.path.join('outputs', 'temp_latents_{}.npy'.format(kwargs["img_name"])), self.sess.run(vars_to_optimize))
                # print(self.sess.run(self.ref_dlatents)[0])
                # print(self.sess.run(self.dlatents)[0])

            if i % 5 == 0:
                # Generate images from found dlatents and save them
                self.generate_images(**kwargs, i=i)

                # # Visualize each Loss
                # print("LOSS1", self.sess.run(1e-4/12*self.loss_1))
                # print("LOSS2", self.sess.run(1e-5*self.loss_2))
                # print("LOSS3", self.sess.run(3e-4*self.loss_3))
                # print("LOSS4", self.sess.run(1e-3*self.loss_4))

            _, loss = self.sess.run([min_op, self.loss])
            yield loss