from PedroScripts.classes import *
from compress import *
import cv2
from tqdm import tqdm


'''anchorPath = "testImages/frame12.png"
targetPath = "testImages/frame13.png"
est = Estimador(anchorPath,targetPath)'''

output_file = 'video_em_frames/comprimido13.mp4'
output_folder = 'video_em_frames/'
video = cv2.VideoCapture('video_menor.mp4')
n_pFrames = 4
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

videofps = video.get(cv2.CAP_PROP_FPS)
videoframecount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

#Lendo o primeiro frame para pegar alguns metadados
ok, i_frame = video.read()
estimador = Estimador(i_frame,i_frame)
height, width = estimador.frameInicial.shape
#cv2.imwrite(f'{output_folder}{4*"0"}i.png',i_frame)


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(filename=output_file, fourcc=fourcc, fps=videofps, frameSize=(width, height), isColor=0)

# Escreve primeiro frame do vídeo
out.write(estimador.frameInicial)
for i in tqdm(range(1,videoframecount-1)):
    ok, cur_frame = video.read()
    if i%n_pFrames == 0:
        i_frame = cur_frame
        estimador = Estimador(i_frame,cur_frame)
        cv2.imwrite(f'{output_folder}{i:04}i.jpeg',estimador.frameInicial)
        out.write(estimador.frameInicial.astype(np.uint8))
        cv2.imshow('janela', estimador.frameInicial)
    else:
        estimador = Estimador(i_frame,cur_frame)
        frameEstimado = estimador.construirFrameEstimado()
        residual = estimador.getResidual()
        
        compressor = Compressor(residual)
        comprimida = compressor.comprimir()

        reconstruida = estimador.reconstuirImagem(comprimida, frameEstimado)
        cv2.imwrite(f'{output_folder}{i:04}p.jpeg',reconstruida)
        out.write(reconstruida.astype(np.uint8))

cv2.destroyAllWindows()
out.release()








######################################################################################################

from tqdm import tqdm
import cv2
import os

# Arguments
dir_path = 'hastes/'
output = 'hastes2.mp4'

images = sorted(os.listdir(dir_path))

# Determine the width and height from the first image
image_path = os.path.join(dir_path, images[0])
print(image_path)
frame = cv2.imread(image_path)
cv2.imshow('video',frame)
height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 30.0, (width, height))

for image in tqdm(images):

    image_path = os.path.join(dir_path, image)
    frame = cv2.imread(image_path)

    out.write(frame) # Write out frame to video

    cv2.imshow('video',frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output))