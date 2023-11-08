import cv2
import numpy as np
import scipy
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
from skimage import io, util


image_path = 'lena.jpg'

image = io.imread(image_path, as_gray=True)

image = image.astype(float)
print(image.max())
print(image.min())

#  K = 10
#  h = np.zeros((K, K))
#  h[K//2,:] = np.ones(K) / K


#  h = np.zeros((2,2))
#  h[0,0] = 1
#  h[1,1] = -1
h = np.zeros((3,3))
h[:,0] = 1/4
h[:,2] = -1/4
h[1,0] = 1/2
h[1,2] = -1/2

#  h = np.zeros((5,5))

#  h[:,0] = 1/11 
#  h[0,0] = 0 
#  h[4,0] = 0 
#  h[:,4] = -1/11 
#  h[0,4] = 0 
#  h[4,4] = 0 
#  h[:,1] = 2/11
#  h[0,1] = 1/11
#  h[4,1] = 1/11
#  h[:,3] = -2/11
#  h[0,3] = -1/11
#  h[4,3] = -1/11

print(h)

g = convolve(image, h, mode='wrap')
print(g.max())
print(g.min())

g = np.clip(g, a_min= 0, a_max=255)
g = g/255
g = util.img_as_ubyte(g)

io.imshow(g)
plt.show()

io.imsave("boarder.png",g)
