from pickle import TRUE
import shutil
from classes import *
from compress import *
import cv2
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import glob
import os

TAMANHO_BLOCO_COMPRESSAO = 20
TAMANHO_BLOCO_ESTIMACAO = 20
# FATOR_COMPRESSAO = 50
TAMANHO_AREA_BUSCA = 7
NUMERO_P_FRAMES = 4

VIDEOS_FOLDER = './videos/'
OUTPUT_FOLDER = './results/'
VIDEOS_TO_RUN = glob.glob(f'{VIDEOS_FOLDER}*.mp4')
GENERATE_FINAL_VIDEO = False

try:
    os.mkdir(OUTPUT_FOLDER)
except:
    pass

print('Running algortihm to following videos:',VIDEOS_TO_RUN)

for videoPath in VIDEOS_TO_RUN:
    for FATOR_COMPRESSAO in range(10,100, 10):
        video = cv2.VideoCapture(videoPath)

        frameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        videoFps = video.get(cv2.CAP_PROP_FPS)
        videoFrameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Lendo o primeiro frame para pegar alguns metadados
        ok, i_frame = video.read()
        estimador = Estimador(i_frame, i_frame, TAMANHO_BLOCO_ESTIMACAO, TAMANHO_AREA_BUSCA)
        height, width = estimador.frameInicial.shape

        videoNameOut = videoPath.replace('.mp4', '').replace(f'{VIDEOS_FOLDER}', '')
        videoOutput = OUTPUT_FOLDER + videoNameOut
        formatedTargetOut = videoNameOut + f' FC({FATOR_COMPRESSAO})'

        print(f'{videoOutput}/{formatedTargetOut}')

        plt.plot([0, 1, 2, 3, 4], [0, 3, 5, 9, 11])
        plt.xlabel('Months')
        plt.ylabel('Books Read')
        plt.savefig(f'{videoOutput}/resultPSNR.png')

        try:
            shutil.rmtree(f'{videoOutput}/{formatedTargetOut}')
        except:
            pass

        try:
            os.mkdir(videoOutput)
        except:
            pass

        try:
            os.mkdir(f'{videoOutput}/{formatedTargetOut}')
        except:
            pass

        cv2.imwrite(f'{videoOutput}/{formatedTargetOut}/{4*"0"}i.png',i_frame)

        for i in tqdm(range(1,2)):
            ok, cur_frame = video.read()
            if i%NUMERO_P_FRAMES == 0:
                i_frame = cur_frame
                estimador = Estimador(i_frame, cur_frame, TAMANHO_BLOCO_ESTIMACAO, TAMANHO_AREA_BUSCA)
                cv2.imwrite(f'{videoOutput}/{formatedTargetOut}/{i:04}i.jpeg',estimador.frameInicial)
            else:
                estimador = Estimador(i_frame, cur_frame, TAMANHO_BLOCO_ESTIMACAO, TAMANHO_AREA_BUSCA)
                frameEstimado = estimador.construirFrameEstimado()
                residual = estimador.getResidual()

                compressor = Compressor(residual, TAMANHO_BLOCO_COMPRESSAO, FATOR_COMPRESSAO)
                comprimida = compressor.comprimir()

                reconstruida = estimador.reconstuirImagem(comprimida, frameEstimado)
                cv2.imwrite(f'{videoOutput}/{formatedTargetOut}/{i:04}p.jpeg',reconstruida)
        
        if GENERATE_FINAL_VIDEO:
            path = formatedTargetOut+'/'

            arquivos = [x for x in os.listdir(path) if x.endswith('.jpeg')]

            for arquivo in arquivos:
                novo  = arquivo.replace('p.jpeg','.jpeg').replace('i.jpeg','.jpeg')
                os.rename(path+arquivo,path+novo)
            
            os.system(f'ffmpeg -framerate 30 -i {videoOutput}/{formatedTargetOut}/%04d.jpeg -codec copy {formatedTargetOut}/output.mkv')


print('Execution sucessfully completed!')