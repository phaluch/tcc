# Code to Perform Block Matching

import numpy as np
import cv2
import os

debug = True

class Estimador:
    def __init__(self, inicial, final, tamBloco=36, tamAreaBusca=7) -> None:
        self.tamBloco = tamBloco
        self.tamAreaBusca = tamAreaBusca
        self.frameInicial, self.frameFinal = self.lendoEPreparandoImagem(inicial, final)
        


    def YCrCb2BGR(self, imagem):
        """
        Converts numpy imagem into from YCrCb to BGR color space
        """
        return cv2.cvtColor(imagem, cv2.COLOR_YCrCb2BGR)

    def BGR2YCrCb(self, imagem):
        """
        Converts numpy imagem into from BGR to YCrCb color space
        """
        return cv2.cvtColor(imagem, cv2.COLOR_BGR2YCrCb)

    def segmentarImagem(self, imagem):
        """
        Calcula quantos macroblocos 'cabem' na imagem
        Isso é importante para conseguir cropar a imagem para um tamanho que vai ser bem recebido pelo resto do algoritmo

        @param imagem: I-Frame
        @return: Quantas linhas e colunas, respectivamente, de macroblocos cabem na imagem
        """
        h, w = imagem.shape
        hSegments = int(h / self.tamBloco)
        wSegments = int(w / self.tamBloco)

        return hSegments, wSegments

    def getCentro(self, x, y):
        """
        Transpõe coordenadas do tipo top-left para coordenadas centrais de um bloco

        @param x, y: Coordenadas x e y, respectivamente, do canto superior esquerdo
        @return: Coordenadas x e y, respectivamente, do centro do macrobloco
        """
        return (int(x + self.tamBloco/2), int(y + self.tamBloco/2))

    def getAreaDeBuscaInicial(self, x:int, y:int, frameInicial:np.ndarray) -> np.ndarray:
        """
        Retorna fatia do frame inicial correspondente ao macrobloco na imagem final, 
        adicionando um padding de tabAreaBusca pixels em cada direção.

        @param x, y: Coordenadas do canto superior esquerdo no macrobloco atual do frame final
        @param frameInicial: Frame inicial
        @return: ndarray contendo área de busca
        """
        h, w = frameInicial.shape
        cx, cy = self.getCentro(x, y)

        sx = max(0, cx-int(self.tamBloco/2)-self.tamAreaBusca) # ensure search area is in bounds
        sy = max(0, cy-int(self.tamBloco/2)-self.tamAreaBusca) # and get top left corner of search area

        # slice inicialframe within bounds to produce inicialsearch area
        areaBuscaInicial = frameInicial[sy:min(sy+self.tamAreaBusca*2+self.tamBloco, h), sx:min(sx+self.tamAreaBusca*2+self.tamBloco, w)]

        return areaBuscaInicial

    def getMacroblocoFrameInicial(self, p, areaInicial, macroblocoFinal):
        """
        Usa a área de busca para retornar o macrobloco no frame inicial correspondente ao macrobloco no frame final

        @param p: x,y Coordenadas do CENTRO do macrobloco escolhido
        @param areaInicial: Área de busca no frame inicial
        @param macroblocoFinal: macrobloco do frame final, para comparação
        @return: macrobloco que foi removido da área de busca, centrado em p
        """
        px, py = p # coordinates of macroblock center
        px, py = px-int(self.tamBloco/2), py-int(self.tamBloco/2) # get top left corner of macroblock
        px, py = max(0,px), max(0,py) # ensure macroblock is within bounds

        macroblocoInicial = areaInicial[py:py+self.tamBloco, px:px+self.tamBloco] # retrive macroblock from inicialsearch area

        try:
            assert macroblocoInicial.shape == macroblocoFinal.shape # must be same shape

        except Exception as e:
            print(e)
            print(f"ERROR - ABLOCK SHAPE: {macroblocoInicial.shape} != TBLOCK SHAPE: {macroblocoFinal.shape}")

        return macroblocoInicial

    def getMAD(self, blocoA, blocoB):
        """
        Calcula o Mean Absolute Difference entre duas matrizes
        """
        return np.sum(np.abs(np.subtract(blocoA, blocoB)))/(blocoA.shape[0]*blocoA.shape[1])

    def getMelhorBloco(self, macroblocoFinal, areaInicial): #3 Step Search
        """
        Implemented 3 Step Search. Read about it here: https://en.wikipedia.org/wiki/Block-matching_algorithm#Three_Step_Search

        @param macroblocoFinal: macrobloco do frame final
        @param areaInicial: Área de busca no frame inicial
        @return: macrobloco contido na área de busca que apresentou a menor MAD
        """
        step = 4
        ah, aw = areaInicial.shape
        acy, acx = int(ah/2), int(aw/2) # get center of inicialsearch area

        minMAD = float("+inf")
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

            for p in range(len(pointList)):
                macroblocoInicial = self.getMacroblocoFrameInicial(pointList[p], areaInicial, macroblocoFinal) # get inicialmacroblock
                MAD = self.getMAD(macroblocoFinal, macroblocoInicial) # determine MAD
                if MAD < minMAD: # store point with minimum mAD
                    minMAD = MAD
                    minP = pointList[p]

            step = int(step/2)

        px, py = minP # center of inicialblock with minimum MAD
        px, py = px - int(self.tamBloco / 2), py - int(self.tamBloco / 2) # get top left corner of minP
        px, py = max(0, px), max(0, py) # ensure minP is within bounds
        matchBlock = areaInicial[py:py + self.tamBloco, px:px + self.tamBloco] # retrieve best macroblock from inicialsearch area

        return matchBlock



    def construirFrameEstimado(self):
        """
        Itera sobre todos os macroblocos do frameFinal fazendo a estimação e compensação em relação ao frameInicial,
        e cria o frameEstimado a partir da realocação dos macroblocos.
        @return: Frame predito por estimação e compensação de movimento
        """
        h, w = self.frameInicial.shape
        hSegments, wSegments = self.segmentarImagem(self.frameInicial)


        frameEstimado = np.ones((h, w))*255
        bcount = 0
        for y in range(0, int(hSegments*self.tamBloco), self.tamBloco):
            for x in range(0, int(wSegments*self.tamBloco), self.tamBloco):
                bcount+=1
                macroblocoFinal = self.frameFinal[y:y+self.tamBloco, x:x+self.tamBloco] #get current macroblock

                areaDeBuscaInicial = self.getAreaDeBuscaInicial(x, y, self.frameInicial) #get inicialsearch area

                #print("AnchorSearchArea: ", areaDeBuscaInicial.shape)

                macroblocoInicial = self.getMelhorBloco(macroblocoFinal, areaDeBuscaInicial) #get best inicialmacroblock
                frameEstimado[y:y+self.tamBloco, x:x+self.tamBloco] = macroblocoInicial #add inicialblock to estimado frame

                #cv2.imwrite("OUTPUT/estimadotestFrame.png", estimado)
                #print(f"ITERATION {bcount}")

        #cv2.imwrite("OUTPUT/estimadotestFrame.png", estimado)

        #time.sleep(10)

        assert bcount == int(hSegments*wSegments) #check all macroblocks are accounted for

        self.frameEstimado = frameEstimado
        return frameEstimado

    def getResidual(self):
        """Create residual frame from inicial,frame - frameEstimado frame"""
        self.frameResidual = np.subtract(self.frameFinal, self.frameEstimado)
        return self.frameResidual

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>.
    # Image Compression
    # return: residual_loss
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>


    def reconstuirImagem(self, frameResidual, frameEstimado):
        """
        Soma dois frames
        """
        return np.add(frameResidual, frameEstimado)

    def showImages(self, *kwargs):
        """
        Mostra imagens
        """
        for k in range(len(kwargs)):
            cv2.imshow(f"Image: {k}", k)
            cv2.waitKey(-1)

    def getResidualMetric(self, residualFrame):
        """
        Calculate residual metric from average of sum of absolute residual values in residual frame
        """
        return np.sum(np.abs(residualFrame))/(residualFrame.shape[0]*residualFrame.shape[1])

    def lendoEPreparandoImagem(self, inicial, final):

        if isinstance(inicial, str) and isinstance(final, str):
            frameInicial = self.BGR2YCrCb(cv2.imread(inicial))[:, :, 0] # get luma component
            frameFinal = self.BGR2YCrCb(cv2.imread(final))[:, :, 0] # get luma component

        elif isinstance(inicial, np.ndarray) and isinstance(final, np.ndarray):
            frameInicial = self.BGR2YCrCb(inicial)[:, :, 0] # get luma component
            frameFinal = self.BGR2YCrCb(final)[:, :, 0] # get luma component

        else:
            raise ValueError

        #resize frame to fit segmentation
        hSegments, wSegments = self.segmentarImagem(frameInicial)
        frameInicial = cv2.resize(frameInicial, (int(wSegments*self.tamBloco), int(hSegments*self.tamBloco)))
        frameFinal = cv2.resize(frameFinal, (int(wSegments*self.tamBloco), int(hSegments*self.tamBloco)))

        #if debug:
            #print(f"A SIZE: {frameInicial.shape}")
            #print(f"T SIZE: {frameFinal.shape}")


        return (frameInicial, frameFinal)

def main(frameInicial, frameFinal, outfile="OUTPUT", saveOutput=False, tamBloco = 16, tamAreaBusca=8):
    """
    Calculate residual frame and metric along with other artifacts
    @param inicial: file path of I-Frame or I-Frame
    @param final: file path of Current Frame or Current Frame
    @return: residual metric
    """

    estimador = Estimador(frameInicial, frameFinal, tamBloco, tamAreaBusca)

    frameInicial, frameFinal = lendoEPreparandoImagem(frameInicial, frameFinal, tamBloco) #processes frame or filepath to frame

    estimadoFrame = construirFrameEstimado(frameInicial, frameFinal, tamBloco)
    residualFrame = getResidual(frameFinal, estimadoFrame)
    naiveResidualFrame = getResidual(frameInicial, frameFinal)
    reconstructTargetFrame = getReconstructTarget(residualFrame, estimadoFrame)
    #showImages(frameFinal, estimadoFrame, residualFrame)

    residualMetric = getResidualMetric(residualFrame)
    naiveResidualMetric = getResidualMetric(naiveResidualFrame)

    rmText = f"Residual Metric: {residualMetric:.2f}"
    nrmText = f"Naive Residual Metric: {naiveResidualMetric:.2f}"

    isdir = os.path.isdir(outfile)
    if not isdir:
        os.mkdir(outfile)

    if saveOutput:
        cv2.imwrite(f"{outfile}/frameFinal.png", frameFinal)
        cv2.imwrite(f"{outfile}/estimadoFrame.png", estimadoFrame)
        cv2.imwrite(f"{outfile}/residualFrame.png", residualFrame)
        cv2.imwrite(f"{outfile}/reconstructTargetFrame.png", reconstructTargetFrame)
        cv2.imwrite(f"{outfile}/naiveResidualFrame.png", naiveResidualFrame)
        resultsFile = open(f"{outfile}/results.txt", "w"); resultsFile.write(f"{rmText}\n{nrmText}\n"); resultsFile.close()

    print(rmText)
    print(nrmText)

    return residualMetric, residualFrame

if __name__ == "__main__":
    pass
    """
    pathInicial = "testImages/personFrame1.png"
    pathDestino = "testImages/personFrame2.png"
    main(pathInicial, pathDestino)
    """
