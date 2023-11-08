from re import sub
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from skimage import io, util
import warnings


#  warnings.filterwarnings("ignore", category=ConvergenceWarning)




video_path = 'test.mp4'
n_gaussians = 4
weight_threshold = 0.75
prob_threshold = 6
#  frameCount = 300

cap = cv2.VideoCapture(video_path)
#  frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameCount = 300
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


video = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))


fc = 0
ret = True

while (fc < frameCount and ret):
    ret, video[fc] = cap.read()
    fc += 1


cap.release()
#  image = video[500]
#  image_new = np.zeros((frameHeight,frameWidth))

print(video.shape)
new_video = np.zeros_like(video)


for i in range(frameHeight):
    print(i)
    for j in range(frameWidth):
        sub_array = video[:,i, j,:]
        gmm = GaussianMixture(n_components=n_gaussians, covariance_type='diag')
        gmm.fit(sub_array)

        #  select the backgrounds:
        weights = gmm.weights_
        #  print(weights)
        sorted_indices = np.argsort(weights)[::-1]

        selected_indices = []
        cumulative_weight = 0

        for k in sorted_indices:
            cumulative_weight += weights[k]
            selected_indices.append(k)
            if cumulative_weight >= weight_threshold:
                break

        #  front = 0
        front = np.zeros(frameCount)
        front = front + 1

        #  for k in selected_indices:
        #      if front == 0:
        #          bias = np.abs(image - gmm.means_[k])
        #          bias_r = bias / gmm.covariances_[k]
        #          if np.max(bias_r < prob_threshold):
        #              front = 1
        

        #  if front == 1:
        #      image_new[i][j] = 1

        for k in selected_indices:
            for r in range(frameCount):
                if front[r] == 1:
                    bias = np.abs(sub_array[r] - gmm.means_[k])
                    bias_r = bias / gmm.covariances_[k]
                    if np.max(bias_r < prob_threshold):
                        front[r] = 0
        

        for r in range(frameCount):
            if front[r] == 1:
                new_video[r][i][j] = [255,255,255]
    #  exit()


#  io.imsave("restored.png",image_new)


frame_size =  (frameWidth, frameHeight)
frame_rate = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
out = cv2.VideoWriter('output_video.mp4', fourcc, frame_rate, frame_size)

for frame in new_video:
    out.write(frame)


out.release()

np.save('my_array.npy', new_video)
