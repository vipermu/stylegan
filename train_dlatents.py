import os
import argparse
import pickle
from tqdm import tqdm
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel

URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  # karras2019stylegan-ffhq-1024x1024.pkl

def main():
    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')

    # Input and Output directories
    parser.add_argument('--src_dir', default='./images', help='Directory with images for encoding')
    parser.add_argument('--out_dir', default='./outputs', help='Directory for storing generated images')

    # Modes of run
    parser.add_argument('--mode', default='dlatents', help='Mode depending on the variables we want to optimize (latents/dlatents)')
    parser.add_argument('--ref_dlatents', default='./outputs/temp_latents.npy', help='Required for the regression of latents given the dlatents')

    # Perceptual model params
    parser.add_argument('--image_size', default=256, help='Size of images for perceptual model', type=int)
    parser.add_argument('--lr', default=0.009, help='Learning rate for perceptual model', type=float)
    parser.add_argument('--iterations', default=10, help='Number of optimization steps for each batch', type=int)

    # Generator params
    parser.add_argument('--randomize_noise', default=False, help='Add noise to dlatents during optimization', type=bool)
    args, other_args = parser.parse_known_args()

    # Store in a list the path for all the images of the source directory
    ref_images = [os.path.join(args.src_dir, x) for x in os.listdir(args.src_dir)]
    ref_images = list(filter(os.path.isfile, ref_images))
    names = [os.path.splitext(os.path.basename(x))[0] for x in ref_images]

    if len(ref_images) == 0:
        raise Exception('%s is empty' % args.src_dir)

    os.makedirs(args.out_dir, exist_ok=True)

    # Initialize generator and perceptual model
    tflib.init_tf()
    
    with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    perceptual_model = PerceptualModel(args.image_size, layer=9)
    
    for idx, ref_img in enumerate(ref_images):
        generator = Generator(Gs_network, randomize_noise=args.randomize_noise, mode=args.mode)
        perceptual_model.build_perceptual_model(generator.generated_image, args.mode, args.ref_dlatents, ref_img)

        perceptual_model.set_reference_images(ref_img)
        op = perceptual_model.optimize(generator.latent_variable, iterations=args.iterations, learning_rate=args.lr,
                                        out_dir = args.out_dir, img_name = names[idx])
        pbar = tqdm(op, leave=False, total=args.iterations)
        for loss in pbar:
            pbar.set_description(names[idx] + ' Loss: %.2f' % loss)
        print(names[idx], ' loss:', loss)

        # Generate images from found dlatents and save them
        generated_images = generator.generate_images()
        generated_dlatent = generator.get_latents()
        if not os.path.exists(args.mode):
            os.mkdir(args.mode)
        np.save(os.path.join(args.mode, names[idx] + '.npy'), generated_dlatent)

if __name__ == "__main__":
    main()
