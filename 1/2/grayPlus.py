import numpy as np
from skimage import io, util
import matplotlib.pyplot as plt

#  load the image in gray scale 
image1 = io.imread("fish.png", as_gray=True,)
image2 = io.imread("feather.png", as_gray=True,)


#  conduct plus operation
image = (image1 + image2) / 2

#  convert the gray scale image from float to [0, 255] original data type
image = util.img_as_ubyte(image)

#  view the composition
io.imshow(image)
plt.show()


#  save the composition
io.imsave("composition.png", image)
