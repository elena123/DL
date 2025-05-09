import os
import random
from PIL import Image
import matplotlib.pyplot as plt

base_folder = '../DATA/clouds/clouds_train'


# https://stackoverflow.com/questions/60443761/displaying-random-images-from-multiple-folders-using-python

def choose_random_images(directory=base_folder):
    images = []
    for folder in os.listdir(directory):
        for image in os.listdir(directory + '/' + folder):
            images.append(os.path.join(directory, folder, image))
    n = 0
    for i in range(6):
        n += 1
        random_img = random.choice(images)
        class_name = os.path.basename(os.path.dirname(random_img))
        imgs = Image.open(random_img)
        sub_plot = plt.subplot(2, 3, n)
        plt.axis('off')
        plt.imshow(imgs)

        sub_plot.set_title(class_name, fontsize=12)
    plt.show()


def main():
    choose_random_images(base_folder)


if __name__ == "__main__":
    main()
