# Script to crop faces from a pic and save them after resizing in 64x64 resolution
from glob import glob
from scipy.misc.pilutil import imread, imsave, imresize
import cv2

faceCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

def crop_and_resize(input_image, outdir):
  # detect face -> crop -> resize -> save
  im = cv2.imread(input_image)
  im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  faces = faceCascade.detectMultiScale(im, scaleFactor=1.5, minNeighbors=5, minSize=(30, 30))
  face_color = None

  for (x,y,w,h) in faces:
      face_color = im[y:y+h, x:x+w]
  
  try:
    small = cv2.resize(face_color, (64, 64))
    file_name = input_image.split('\\')[-1]
    imsave("{}/{}".format(outdir, file_name), small)

  except Exception:
    # if face is not detected  
    im = imread(input_image)
    height, width, color = im.shape
    edge_h = int( round( (height - 108) / 2.0 ) )
    edge_w = int( round( (width - 108) / 2.0 ) )

    cropped = im[edge_h:(edge_h + 108), edge_w:(edge_w + 108)]
    small = imresize(cropped, (64, 64))

    file_name = input_image.split('\\')[-1]
    imsave("{}/{}".format(outdir, file_name), small)  

X = glob("celeb_faces\\*.jpg")
N = len(X)
print(" {} faces found".format(N))

print("Cropping and resizing images")

for i in range(N):
    crop_and_resize(X[i], 'faces')
    if i % 1000 == 0:
        print("{}/{}".format(i, N))

print("Cropped and resized successfully")