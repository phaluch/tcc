from pickle import TRUE
import shutil
from classes import *
from compress import *
import cv2
from tqdm import tqdm
import glob
import os

TAMANHO_BLOCO = 24
FATOR_COMPRESSAO = 95
TAMANHO_AREA_BUSCA = 7
NUMERO_P_FRAMES = 4

VIDEOS_FOLDER = './videos/'
OUTPUT_FOLDER = './results/'
VIDEOS_TO_RUN = glob.glob(f'{VIDEOS_FOLDER}*.mp4')
GENERATE_FINAL_VIDEO = False


def getPartialPSNR(originalImage, finalImage):
    originalImage=originalImage[0:finalImage.shape[0], 0:finalImage.shape[1]]
    diff = originalImage - finalImage
    diffsq = diff**2
    tam = diffsq.shape[0] * diffsq.shape[1]

    MSE = np.sum(diffsq)/tam
    PSNR = (20 * np.log10(255)) - (10 * np.log10(MSE))

    return PSNR

try:
    os.mkdir(OUTPUT_FOLDER)
except:
    pass

print('Running algortihm to following videos:',VIDEOS_TO_RUN)

for videoPath in VIDEOS_TO_RUN:
    video = cv2.VideoCapture(videoPath)

    frameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    videoFps = video.get(cv2.CAP_PROP_FPS)
    videoFrameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Lendo o primeiro frame para pegar alguns metadados
    ok, i_frame = video.read()
    estimador = Estimador(i_frame, i_frame, TAMANHO_BLOCO, TAMANHO_AREA_BUSCA)
    height, width = estimador.frameInicial.shape

    formatedTargetOut = OUTPUT_FOLDER + videoPath.replace('.mp4', '').replace(f'{VIDEOS_FOLDER}', '')

    # try:
        # shutil.rmtree(formatedTargetOut)
    os.mkdir(formatedTargetOut)
    # except:
        # pass

    cv2.imwrite(f'{formatedTargetOut}/{4*"0"}i.png',i_frame)

    for i in tqdm(range(1,videoFrameCount-1)):
        ok, cur_frame = video.read()
        if i%NUMERO_P_FRAMES == 0:
            i_frame = cur_frame
            estimador = Estimador(i_frame, cur_frame, TAMANHO_BLOCO, TAMANHO_AREA_BUSCA)
            # print(getPartialPSNR(cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY),cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)))
            cv2.imwrite(f'{formatedTargetOut}/{i:04}i.jpeg',estimador.frameInicial)
        else:
            estimador = Estimador(i_frame, cur_frame, TAMANHO_BLOCO, TAMANHO_AREA_BUSCA)
            frameEstimado = estimador.construirFrameEstimado()
            residual = estimador.getResidual()
            
            compressor = Compressor(residual, TAMANHO_BLOCO, FATOR_COMPRESSAO)
            comprimida = compressor.comprimir()

            reconstruida = estimador.reconstuirImagem(comprimida, frameEstimado)
            print(getPartialPSNR(cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY),reconstruida))
            cv2.imwrite(f'{formatedTargetOut}/{i:04}poriginal.jpeg',cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY))
            cv2.imwrite(f'{formatedTargetOut}/{i:04}p.jpeg',reconstruida)
    
    if GENERATE_FINAL_VIDEO:
        path = formatedTargetOut+'/'

        arquivos = [x for x in os.listdir(path) if x.endswith('.jpeg')]

        for arquivo in arquivos:
            novo  = arquivo.replace('p.jpeg','.jpeg').replace('i.jpeg','.jpeg')
            os.rename(path+arquivo,path+novo)
        
        os.system(f'ffmpeg -framerate 30 -i {formatedTargetOut}/%04d.jpeg -codec copy {formatedTargetOut}/output.mkv')


print('Execution sucessfully completed!')