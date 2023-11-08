import numpy as np
from skimage import io, util
import matplotlib.pyplot as plt
import math
import cv2
import pickle




image_path = 'lena.jpg'
compress_rate = 0.1
dct_path = 'dct.pkl'
extracted_image_path = 'extracted_image.jpg'




edge_rate =  math.sqrt(compress_rate)
image = io.imread(image_path, as_gray = True)

#  compress
dct = cv2.dct(image.astype(float))


cliped_dct = dct[:int(image.shape[0] * edge_rate) , :int(image.shape[1] * edge_rate)]

with open(dct_path, 'wb') as f:
    pickle.dump(cliped_dct, f)
    f.close()



#  extract
extended_dct =  np.zeros_like(image)
extended_dct = extended_dct.astype(np.float32)
extended_dct[:int(image.shape[0] * edge_rate) , :int(image.shape[1] * edge_rate)] = cliped_dct
extended_dct = extended_dct.astype(np.float32)


extracted_image = cv2.idct(extended_dct)
extracted_image = extracted_image / 255
extracted_image = np.clip(extracted_image, a_min= 0, a_max=1)

extracted_image = util.img_as_ubyte(extracted_image)

io.imshow(extracted_image)
plt.show()
io.imsave(extracted_image_path,extracted_image)
