import cv2
import numpy as np
import scipy.fftpack as fp
from scipy import fftpack
from skimage import io, util
import matplotlib.pyplot as plt


image_path = 'lena.png'

image = io.imread(image_path, as_gray=True,)

#  image = util.img_as_ubyte(image)

F1 = fp.fft2((image).astype(float))
F2 = fp.fftshift(F1)
plt.figure(figsize=(10,10))
plt.imshow((20*np.log10(0.1+F2)).astype(int),cmap=plt.cm.gray)
plt.show()

mask = np.zeros_like(F2)
#  mask = mask + 1
(width, height) = F2.shape
print(F2.shape)

#  n0 = 220
#  n1 = 173
#  m1 = 175
#  n2 = 122
#  m2 = 124

#  mask[int(width/2) - n0:int(width/2) + n0 + 1, int(height/2) - n0:int(height/2) + n0 + 1] = 1
#  mask[int(width/2) - m1:int(width/2) + m1 + 1, int(height/2) - m1:int(height/2) + m1 + 1] = 0
#  mask[int(width/2) - n1:int(width/2) + n1 + 1, int(height/2) - n1:int(height/2) + n1 + 1] = 1
#  mask[int(width/2) - m2:int(width/2) + m2+ 1, int(height/2) - m2:int(height/2) + m2+ 1] = 0
#  mask[int(width/2) - n2:int(width/2) + n2 + 1, int(height/2) - n2:int(height/2) + n2 + 1] = 1

avg1 = np.mean(F2)
radius = 2

r1 = 123
r2 = 174

mask = mask + 1
mask[int(width/2) - radius:int(width/2) + radius, int(height/2) - r2 - radius :int(height/2) -r2 + radius] = 0
mask[int(width/2) - radius:int(width/2) + radius, int(height/2) + r2 - radius:int(height/2) +r2  + radius] = 0
mask[ int(width/2) - r2  - radius:int(width/2) -r2 + radius ,int(height/2) - radius:int(height/2) + radius] = 0
mask[ int(width/2) + r2  - radius:int(width/2) +r2 +radius,int(height/2) - radius:int(height/2) + radius]= 0

mask[int(width/2) - r1  - radius:int(width/2) -r1 + radius, int(height/2) - r1 - radius :int(height/2) -r1 +radius] = 0
mask[int(width/2) + r1  - radius:int(width/2) +r1 + radius, int(height/2) - r1 - radius :int(height/2) -r1 + radius] = 0
mask[int(width/2) - r1  - radius:int(width/2) -r1 + radius, int(height/2) + r1 - radius :int(height/2) +r1 + radius] = 0
mask[int(width/2) + r1  - radius:int(width/2) +r1 + radius, int(height/2) + r1 - radius :int(height/2) +r1 + radius] = 0

F2 = F2 * mask

plt.figure(figsize=(10,10))
plt.imshow((20*np.log10(0.1+F2)).astype(int),cmap=plt.cm.gray)
plt.show()


image_new = fp.ifft2(fftpack.ifftshift(F2)).real
image_new = np.clip(image_new, a_min= 0, a_max=1)

print(np.max(image_new))
print(image_new)


image_new = util.img_as_ubyte(image_new)


io.imshow(image_new)
plt.show()
io.imsave("filtered.png",image_new)
