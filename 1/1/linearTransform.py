import numpy as np
from skimage import io, util
import matplotlib.pyplot as plt

#  load the image in gray scale 
image = io.imread("fish.png", as_gray=True,)

#  convert the gray scale image from float to [0, 255] original data type
image = util.img_as_ubyte(image)


#  view the gray scaled image
io.imshow(image)
plt.show()

#  save the gray scale image
io.imsave("grayscale.png", image)

#linear transformation, invert the intensity
image = image.max() - image

#  view the transformed image
io.imshow(image)
plt.show()


#  save the transformed image
io.imsave("transformed.png", image)
