import numpy as np
from skimage import io, util
import huffman 
import csv
import struct
import matplotlib.pyplot as plt



#  write the image shape and encoded image
def encoder(image, encoding_dict, file_path):
    with open(file_path, 'bw') as f:
        f.write(image.shape[0].to_bytes(4, byteorder = 'little'))
        f.write(image.shape[1].to_bytes(4, byteorder = 'little'))
        to_write = ''
        for x in range(image.shape[0]):
           for y in range(image.shape[1]):
               to_write = to_write + encoding_dict[image[x][y]]
               while(len(to_write) >= 8):
                   unit = to_write[:8]
                   to_write = to_write[8:]
                   bin_data = struct.pack('B',int(unit, 2))
                   f.write(bin_data)
        while len(to_write) < 8:
            to_write = to_write + '0'
        bin_data = struct.pack('B',int(to_write, 2))
        f.write(bin_data)
        f.close()


#  read the huffman file and decode it return the decoded image ndarray
def decoder(decoding_dict, file_path):
    height = 0
    width = 0
    with open(file_path, 'br') as f:
        tmp = f.read(4)
        height = int.from_bytes(tmp, byteorder = 'little')
        tmp = f.read(4)
        width = int.from_bytes(tmp, byteorder = 'little')
        print(height, width)
        size = height * width
        readed_size = 0
        to_decode = ''
        value_list = list()
        print(size)
        #  user_input = input("Press Enter to continue...")
        while (readed_size < size):
            #  user_input = input("Press Enter to continue...")
            new_byte = f.read(1)
            additional_str = bin(int.from_bytes(new_byte, byteorder='little'))
            to_decode = to_decode +  additional_str[2:].zfill(8)
            i = 1
            while i <= len(to_decode):
                if decoding_dict.get(to_decode[:i]) is None:
                    i += 1
                else:
                    value_list.append(decoding_dict[to_decode[:i]])
                    readed_size += 1
                    to_decode = to_decode[i:]
                    i = 1

    image = np.zeros((height, width))
    for x in range(height):
        for y in range(width):
            image[x][y] = value_list[x*width + y]
    image = image.astype(int)
    return image





#  write the coding table in a csv file, keys at first col and coding at the second
def writeCodingTable(table_path, encoding_dict):
    with open(table_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in encoding_dict.items():
            writer.writerow([key,value])
        csv_file.close()


#  return the decoding dict
def readCodingTable(table_path):
    decoding_dict = dict()
    with open(table_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            decoding_dict[row[1]] = int(row[0])
        csv_file.close()   
    return decoding_dict
        




if __name__ == "__main__":
    image_path = 'lena.jpg'
    coding_table_path = 'coding_table.csv'
    encoded_image_path = 'encoded_file.b'
    table_path = 'huffmanTable.csv'


    #  read image
    image = io.imread(image_path, as_gray = True)

    #  generate coding table
    values, counts = np.unique(image, return_counts= True)
    data_info = list(zip(values, counts))
    encoding_dict = huffman.codebook(data_info)

    #  save coding table
    writeCodingTable(table_path=table_path, encoding_dict=encoding_dict)

    #  encoding image and save it
    encoder(image=image, encoding_dict=encoding_dict, file_path = encoded_image_path)

    #  read coding table
    decoding_dict = readCodingTable(table_path=table_path)
    #  read encoded image and decode it
    decoded_image = decoder(decoding_dict=decoding_dict, file_path=encoded_image_path)

    #  check
    decoded_image = decoded_image / 255
    decoded_image = util.img_as_ubyte(decoded_image)
    io.imshow(decoded_image)
    plt.show()

    io.imsave("decoded.png", decoded_image)

