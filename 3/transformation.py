import cv2
import numpy as np


image_path = 'lena.png'

img = cv2.imread(image_path)
rows, cols = img.shape[:2]
print(rows, cols)


#  affine
src_points = np.float32([[0,0], [cols-1,0], [0, rows-1], [cols-1, rows -1]])
dst_points_affine = np.float32([[int(0.33*rows),int(0.66*rows)-1], [int(cols-1), 0], [int(0.33*rows), rows -1], [cols -1, int(0.33*rows) -1]])
projective_matrix_affine = cv2.getPerspectiveTransform(src_points, dst_points_affine)
img_output_affine = cv2.warpPerspective(img, projective_matrix_affine, (cols, rows))


# projective
src_points = np.float32([[0,0], [cols-1,0], [0, rows-1], [cols-1, rows -1]])
dst_points_projective = np.float32([[0,int(0.66*rows)-1], [int(cols-1), 0], [0, rows -1], [int(0.66*cols), rows -1]])
projective_matrix_projective = cv2.getPerspectiveTransform(src_points, dst_points_projective)
img_output_projective = cv2.warpPerspective(img, projective_matrix_projective, (cols, rows))


cv2.imwrite('affined.png', img_output_affine)
cv2.imwrite('projectived.png', img_output_projective)
