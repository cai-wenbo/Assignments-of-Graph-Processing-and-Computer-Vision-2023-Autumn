import numpy as np
from scipy import ndimage
from skimage import io, util
import matplotlib.pyplot as plt







def main():
    image_path = '1.png'
    image = io.imread(image_path, as_gray = True)
    print(image.shape)
    print(image)


    new_image = image.copy()
    mask = ndimage.generate_binary_structure(2,1)

    mask1 = np.zeros((3, 3), dtype=bool)
    mask1[2, 0:3] = True

    mask2 = np.zeros((3, 3), dtype=bool)
    mask2[0:3, 1] = True

    new_image = ndimage.grey_erosion(new_image, size = (3, 3), footprint = mask1)
    new_image = ndimage.grey_dilation(new_image, size = (3, 3), footprint = mask1)



    new_image = ndimage.grey_erosion(new_image, size = (3, 3), footprint = mask2)
    new_image = ndimage.grey_dilation(new_image, size = (3, 3), footprint = mask2)





    new_image = (image + new_image) / 2

    new_image = ndimage.grey_erosion(new_image, size = (3, 3), footprint = mask)
    new_image = ndimage.grey_dilation(new_image, size = (3, 3), footprint = mask)



    new_image = ndimage.grey_dilation(new_image, size = (3, 3), footprint = mask)
    new_image = ndimage.grey_erosion(new_image, size = (3, 3), footprint = mask)

    
    new_image = util.img_as_ubyte(new_image)
    io.imsave("filtered.png",new_image)





if __name__ == "__main__":
    main()
