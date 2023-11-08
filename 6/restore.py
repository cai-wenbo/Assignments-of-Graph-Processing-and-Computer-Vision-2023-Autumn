import cv2
import math
import numpy as np
import scipy
from scipy import fftpack
import scipy.fftpack as fp
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
from skimage import io, util


def fill_anti_diagonal(m, val):
    n = m.copy()
    np.fill_diagonal(n, val)
    n = np.flipud(n)
    return n
    
    


image_path = 'blurred.jpg'
#  just guess the shape of the degradation model.
k=18

image = io.imread(image_path, as_gray=True)


h = np.zeros((k,k))

h = fill_anti_diagonal(h, 1/k)

H = np.zeros_like(image)
H[(image.shape[0]-h.shape[0])//2:(image.shape[0]-h.shape[0])//2+h.shape[0],(image.shape[1]-h.shape[1])//2:(image.shape[1]-h.shape[1])//2+h.shape[1]] = h

fH1 = fp.fft2((H).astype(float))
fH2 = fp.fftshift(fH1)

print(np.max(fH2))
#  plt.figure(figsize=(10,10))
#  plt.imshow((20*np.log10(0.1+fH2)).astype(int),cmap=plt.cm.gray)
#  plt.show()

fimage1 = fp.fft2((image).astype(float))
fimage2 = fp.fftshift(fimage1)

print(fH1.shape)
print(fimage1.shape)

threshold = 0.01

fHinv1 = np.zeros_like(fimage1)

for i in range(fHinv1.shape[0]):
    for j in range(fHinv1.shape[1]):
        mag = np.abs(fH1[i,j])
        if mag < threshold:
            fHinv1[i,j] = 0
        else:
            fHinv1[i,j] = 1.0 / (fH1[i,j])
        #  fHinv1[i,j]=1
        


fHinv2 = fp.fftshift(fHinv1)


#  plt.figure(figsize=(10,10))
#  plt.imshow((20*np.log10(0.1+fHinv2)).astype(int),cmap=plt.cm.gray)
#  plt.show()

#  plt.figure(figsize=(10,10))
#  plt.imshow((20*np.log10(0.1+fimage2)).astype(int),cmap=plt.cm.gray)
#  plt.show()

F = fimage1 * fHinv1

print(F.shape)

#  plt.figure(figsize=(10,10))
#  plt.imshow((20*np.log10(0.1+F)).astype(int),cmap=plt.cm.gray)
#  plt.show()


#  image_new = fp.ifft2(fftpack.ifftshift(F)).real
#  image_new = np.clip(image_new, a_min= 0, a_max=1)
#  io.imshow(image_new)
#  plt.show()

image_new = fp.ifft2(F).real
image_new = np.clip(image_new, a_min= 0, a_max=1)

image_new = util.img_as_ubyte(image_new)

top_half = image_new[:247]
bottom_half = image_new[247:]

image_new = np.concatenate((bottom_half, top_half))

right_half = image_new[:,264:]
left_half = image_new[:,:264]

image_new = np.concatenate((right_half, left_half),axis = 1)



#  io.imshow(image_new)
#  plt.show()

io.imsave("restored.png",image_new)
#  fH
