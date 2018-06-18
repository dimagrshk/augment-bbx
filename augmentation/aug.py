from collections import namedtuple

import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from skimage.io import imsave
import cv2

ia.seed(1)

Rectangle = namedtuple('Rectangle', ['x', 'y', 'w', 'h'])

image = ia.quokka(size=(256, 256))
rect = Rectangle(65, 100, 200, 150)

def crop(img: np.ndarray, r: Rectangle) -> np.ndarray:
    return img[r.y:r.y+r.h, r.x:r.x+r.w]

def augment(img, bb, sequence):
    crop_image = crop(img, bb)
    for i, s in enumerate(sequence):
        aug_img = s.augment_image(crop_image)
        imsave(f'augmented/{i}.jpg', aug_img)
        for _ in range(10):
            imsave(f'augmented/{i}{_}_rotate.jpg', rotate_aug.augment_image(aug_img))


rotate_aug = iaa.Affine(rotate=(-0.8, 0.8))

bbs = ia.BoundingBoxesOnImage([
    ia.BoundingBox(x1=rect.x, y1=rect.y, x2=rect.w, y2=rect.h)
], shape=image.shape)

seq = iaa.Sequential([
    iaa.GaussianBlur((0, 3.0)),
    iaa.AverageBlur(k=(2, 7)),
    iaa.MedianBlur(k=(3, 11)),
    iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ])),
    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
    iaa.Dropout((0.01, 0.2), per_channel=0.5),
    iaa.Invert(0.05, per_channel=True),
    iaa.Add((-10, 50), per_channel=0.5),
    iaa.Fliplr(0.5, name="Flipper"),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.2), per_channel=0.5)
], random_order=True)

seq_det = seq.to_deterministic()


if __name__ == '__main__':
    augment(image, rect, seq_det)
    image_before = bbs.draw_on_image(image, thickness=2)
