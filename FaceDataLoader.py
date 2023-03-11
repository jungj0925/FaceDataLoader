import imgaug.augmenters as iaa
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import numpy as np
import cv2
import os

class FaceDataLoader:
    def __init__(self, image, keypoints):
        self.image = image
        self.keypoints = keypoints

    def augment(image, keypoints):
        image = cv2.imread(image)
        # Load the keypoints text file
        with open(keypoints, 'r') as f:
            keypoints_str = f.readline().strip()  # assuming one line of keypoints in the file
        # Split the string into individual coordinate values
        keypoints_list = [float(x) for x in keypoints_str.split()]
        # Reshape the list into a numpy array
        keypoints = np.array(keypoints_list).reshape((-1, 2))
        keypoints = [Keypoint(x=kp[0], y=kp[1]) for kp in keypoints]
        keypoints = KeypointsOnImage(keypoints, shape=image.shape)
        seq = iaa.Sequential([
            iaa.Fliplr(0.5), # horizontally flip 50% of the images
            iaa.Affine(rotate=(-45, 45)), # apply a random rotation between -45 to 45 degrees
            iaa.AdditiveGaussianNoise(scale=(0, 0.1*255)), # add random gaussian noise with a scale of 0 to 0.1 times 255
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            )
        ])
        image_aug, keypoints_aug = seq(image=image, keypoints=keypoints)
        image_aug = keypoints_aug.draw_on_image(image_aug)
        augmented_keypoints = keypoints_aug.to_xy_array()

        # Create the file if it doesn't exist
        if not os.path.exists('./Output/augmented_keypoints.txt'):
            open('./Output/augmented_keypoints.txt', 'w').close()

        # Save the coordinates to the text file
        with open('./Output/augmented_keypoints.txt', 'w') as f:
            for kp in augmented_keypoints:
                f.write('{} {} '.format(kp[0], kp[1]))
        cv2.imwrite('./Output/augmented_image.jpg', image_aug)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

img_url = './9_Press_Conference_Press_Conference_9_281.jpg'
key_url = './keypoints.txt'
FaceDataLoader.augment(image=img_url, keypoints=key_url)