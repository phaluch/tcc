from pickle import TRUE
import shutil
from classes import *
from compress import *
import cv2
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import time
import os
import csv
from joblib import Parallel, delayed

OUTPUT_FOLDER = './results/'
GENERATE_FINAL_VIDEO = True

videoPath = sys.argv[1]

print(videoPath)

try:
    os.mkdir(OUTPUT_FOLDER)
except:
    pass

print('Running algortihm to following video:', videoPath)

def executa(TAMANHO_BLOCO_ESTIMACAO, TAMANHO_AREA_BUSCA, NUMERO_P_FRAMES, FATOR_COMPRESSAO):
    video = cv2.VideoCapture(videoPath)

    frameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    videoFps = video.get(cv2.CAP_PROP_FPS)
    videoFrameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Lendo o primeiro frame para pegar alguns metadados
    ok, i_frame = video.read()
    estimador = Estimador(i_frame, i_frame, TAMANHO_BLOCO_ESTIMACAO, TAMANHO_AREA_BUSCA)
    height, width = estimador.frameInicial.shape

    videoNameOut = videoPath.replace('.mp4', '')
    videoOutput = OUTPUT_FOLDER + videoNameOut
    formatedTargetOut = videoNameOut + f'_FC[{FATOR_COMPRESSAO}]_PF[{NUMERO_P_FRAMES}]_SA[{TAMANHO_AREA_BUSCA}]_TBE[{TAMANHO_BLOCO_ESTIMACAO}]'

    try:
        shutil.rmtree(f'{videoOutput}/outputs/{formatedTargetOut}')
    except:
        pass

    try:
        os.mkdir(videoOutput)
    except:
        pass

    try:
        os.mkdir(f'{videoOutput}/outputs')
    except:
        pass

    try:
        os.mkdir(f'{videoOutput}/mounted_videos')
    except:
        pass

    try:
        os.mkdir(f'{videoOutput}/outputs/{formatedTargetOut}')
    except:
        pass

    f = open(f'{videoOutput}/outputs/{formatedTargetOut}/{formatedTargetOut}.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['frame', 'estimacao', 'compressao', 'reconstrucao'])

    cv2.imwrite(f'{videoOutput}/outputs/{formatedTargetOut}/{4*"0"}i.png',cv.cvtColor(i_frame, cv2.COLOR_BGR2GRAY))

    for i in range(1,int((videoFrameCount-1))):
        ok, cur_frame = video.read()
        if i%NUMERO_P_FRAMES == 0:
            i_frame = cur_frame
            estimador = Estimador(i_frame, cur_frame, TAMANHO_BLOCO_ESTIMACAO, TAMANHO_AREA_BUSCA)
            cv2.imwrite(f'{videoOutput}/outputs/{formatedTargetOut}/{i:04}i.jpeg',estimador.frameInicial)
        else:
            startEstim = time.process_time()

            estimador = Estimador(i_frame, cur_frame, TAMANHO_BLOCO_ESTIMACAO, TAMANHO_AREA_BUSCA)
            frameEstimado = estimador.construirFrameEstimado()
            residual = estimador.getResidual()

            timeEstim = time.process_time() - startEstim

            startCompression = time.process_time()

            compressor = Compressor(residual, 1, FATOR_COMPRESSAO)
            comprimida = compressor.comprimir()

            timeCompression = time.process_time() - startCompression

            startReconstrucao = time.process_time()

            reconstruida = estimador.reconstuirImagem(comprimida, frameEstimado)

            timeReconstrucao = time.process_time() - startReconstrucao

            writer.writerow([i, timeEstim, timeCompression, timeReconstrucao])

            if FATOR_COMPRESSAO == 0:
                cv2.imwrite(f'{videoOutput}/outputs/{formatedTargetOut}/{i:04}p.jpeg',estimador.frameFinal)
            else:
                cv2.imwrite(f'{videoOutput}/outputs/{formatedTargetOut}/{i:04}residual.jpeg',residual)
                cv2.imwrite(f'{videoOutput}/outputs/{formatedTargetOut}/{i:04}p.jpeg',reconstruida)
    
    if GENERATE_FINAL_VIDEO:
        path = f'{videoOutput}/outputs/{formatedTargetOut}/'

        arquivos = [x for x in os.listdir(path) if x.endswith('.jpeg')]

        for arquivo in arquivos:
            novo  = arquivo.replace('p.jpeg','.jpeg').replace('i.jpeg','.jpeg')
            os.rename(path+arquivo,path+novo)
        
        os.system(f'ffmpeg -framerate 30 -i {videoOutput}/outputs/{formatedTargetOut}/%04d.jpeg -codec copy {videoOutput}/mounted_videos/{formatedTargetOut}.mkv')

    return 'Execution sucessfully completed!'
    

resultado = Parallel(n_jobs=3)(delayed(executa)(TAMANHO_BLOCO_ESTIMACAO, TAMANHO_AREA_BUSCA, NUMERO_P_FRAMES, FATOR_COMPRESSAO) for TAMANHO_BLOCO_ESTIMACAO in range(8,17,4) for TAMANHO_AREA_BUSCA in [round(TAMANHO_BLOCO_ESTIMACAO * fator) for fator in [1.0,1.5,2.0]] for NUMERO_P_FRAMES in range(15,0,-2) for FATOR_COMPRESSAO in range(90, 100, 1))

