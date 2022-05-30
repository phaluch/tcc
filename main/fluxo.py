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
from numba import jit

OUTPUT_FOLDER = "results/"
GENERATE_FINAL_VIDEO = True

videoPath = sys.argv[1]

print(videoPath)

try:
    os.mkdir(OUTPUT_FOLDER)
except:
    pass

print('Running algortihm to following video:', videoPath)


@jit
def j_construirFrameEstimado(frameInicial, frameFinal, tamBloco, tamAreaBusca):
    """
    Itera sobre todos os macroblocos do frameFinal fazendo a estimação e compensação em relação ao frameInicial,
    e cria o frameEstimado a partir da realocação dos macroblocos.
    @return: Frame predito por estimação e compensação de movimento
    """
    h, w = frameInicial.shape
    hSegments = int(h / tamBloco)
    wSegments = int(w / tamBloco)

    frameEstimado = np.ones((h, w))*255
    bcount = 0
    for y in range(0, int(hSegments*tamBloco), tamBloco):
        for x in range(0, int(wSegments*tamBloco), tamBloco):
            bcount+=1
            macroblocoFinal = frameFinal[y:y+tamBloco, x:x+tamBloco] #get current macroblock

            cx, cy = (int(x + tamBloco/2), int(y + tamBloco/2))

            sx = max(0, cx-int(tamBloco/2)-tamAreaBusca) # ensure search area is in bounds
            sy = max(0, cy-int(tamBloco/2)-tamAreaBusca) # and get top left corner of search area

            # slice inicialframe within bounds to produce inicialsearch area
            areaInicial = frameInicial[sy:min(sy+tamAreaBusca*2+tamBloco, h), sx:min(sx+tamAreaBusca*2+tamBloco, w)]
            

            #print("AnchorSearchArea: ", areaDeBuscaInicial.shape)
            step = 4
            ah, aw = areaInicial.shape
            acy, acx = int(ah/2), int(aw/2) # get center of inicialsearch area

            minMAD = np.inf
            minP = None

            while step >= 1:
                p1 = (acx, acy)
                p2 = (acx+step, acy)
                p3 = (acx, acy+step)
                p4 = (acx+step, acy+step)
                p5 = (acx-step, acy)
                p6 = (acx, acy-step)
                p7 = (acx-step, acy-step)
                p8 = (acx+step, acy-step)
                p9 = (acx-step, acy+step)
                pointList = [p1,p2,p3,p4,p5,p6,p7,p8,p9] # retrieve 9 search points

                for pl in range(len(pointList)):
                    px, py = pointList[pl] # coordinates of macroblock center
                    px, py = px-int(tamBloco/2), py-int(tamBloco/2) # get top left corner of macroblock
                    px, py = max(0,px), max(0,py) # ensure macroblock is within bounds

                    macroblocoInicial = areaInicial[py:py+tamBloco, px:px+tamBloco] # retrive macroblock from inicialsearch area

                    
                    if macroblocoInicial.shape != macroblocoFinal.shape:
                        raise Exception
                        
                    MAD = np.sum(np.abs(np.subtract(macroblocoFinal, macroblocoInicial)))/(macroblocoFinal.shape[0]*macroblocoFinal.shape[1])
                    # determine MAD
                    if MAD < minMAD: # store point with minimum mAD
                        minMAD = MAD
                        minP = pointList[pl]

                step = int(step/2)

            px, py = minP # center of inicialblock with minimum MAD
            px, py = px - int(tamBloco / 2), py - int(tamBloco / 2) # get top left corner of minP
            px, py = max(0, px), max(0, py) # ensure minP is within bounds
            macroblocoInicial = areaInicial[py:py + tamBloco, px:px + tamBloco] # retrieve best macroblock from inicialsearch area
            
            frameEstimado[y:y+tamBloco, x:x+tamBloco] = macroblocoInicial #add inicialblock to estimado frame

            #cv2.imwrite("OUTPUT/estimadotestFrame.png", estimado)
            #print(f"ITERATION {bcount}")

    #cv2.imwrite("OUTPUT/estimadotestFrame.png", estimado)

    #time.sleep(10)

    assert bcount == int(hSegments*wSegments) #check all macroblocks are accounted for

    frameEstimado = frameEstimado
    return frameEstimado



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

    for i in tqdm(range(1,int((videoFrameCount-1))), desc=formatedTargetOut, colour='CYAN'):
        ok, cur_frame = video.read()
        if i%NUMERO_P_FRAMES == 0:
            i_frame = cur_frame
            estimador = Estimador(i_frame, cur_frame, TAMANHO_BLOCO_ESTIMACAO, TAMANHO_AREA_BUSCA)
            cv2.imwrite(f'{videoOutput}/outputs/{formatedTargetOut}/{i:04}i.jpeg',estimador.frameInicial)
        else:
            startEstim = time.process_time()

            estimador = Estimador(i_frame, cur_frame, TAMANHO_BLOCO_ESTIMACAO, TAMANHO_AREA_BUSCA)

            frameEstimado = j_construirFrameEstimado(estimador.frameInicial, estimador.frameFinal, TAMANHO_BLOCO_ESTIMACAO, TAMANHO_AREA_BUSCA)
            estimador.frameEstimado = frameEstimado
            ############################################################################################################ 
            # frameEstimado = estimador.construirFrameEstimado()
            ########################################################################################
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

executa(8,8,2,0)
# Parallel(n_jobs=-1)(delayed(executa)(TAMANHO_BLOCO_ESTIMACAO, TAMANHO_AREA_BUSCA, NUMERO_P_FRAMES, FATOR_COMPRESSAO) for TAMANHO_BLOCO_ESTIMACAO in range(8,17,4) for TAMANHO_AREA_BUSCA in [round(TAMANHO_BLOCO_ESTIMACAO * fator) for fator in [1.0,1.5,2.0]] for NUMERO_P_FRAMES in range(15,0,-2) for FATOR_COMPRESSAO in range(90, 100, 1))

