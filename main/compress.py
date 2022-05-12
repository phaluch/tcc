import string
from xmlrpc.client import boolean
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm


class Compressor:
    def __init__(self, imagemOriginal: np.ndarray, tamanhoBloco, fatorCompressao):
        self.tamanhoBloco = tamanhoBloco
        self.fatorCompressao = fatorCompressao
        self.imagemOriginal = imagemOriginal
        self.h, self.w = (np.array(self.imagemOriginal.shape[:2])/self.tamanhoBloco * self.tamanhoBloco).astype(int)
        self.blocosV = int(self.h/self.tamanhoBloco)
        self.blocosH = int(self.w/self.tamanhoBloco)
        self.imagemValidada = np.zeros((self.blocosV*self.tamanhoBloco,self.blocosH*self.tamanhoBloco))
        self.imagemValidada[:self.blocosV*self.tamanhoBloco,:self.blocosH*self.tamanhoBloco] = self.imagemOriginal[:self.blocosV*self.tamanhoBloco,:self.blocosH*self.tamanhoBloco]

    def comprimir(self, salvarDisco:bool = False):
        dctizador = DCTizador(self.tamanhoBloco, self.blocosV, self.blocosH)
        pixelKiller = PixelKiller(self.tamanhoBloco, self.blocosV, self.blocosH)
        imagemResDCT = dctizador.aplicarDCT(self.imagemValidada)
        imagemResEliminacao = pixelKiller.eliminarPixels(imagemResDCT, self.fatorCompressao)
        imagemResultante = dctizador.aplicarIDCT(imagemResEliminacao)

        #Utilidades.getMADEntreImagens(self.imagemValidada, imagemResultante, self.blocosV, self.blocosH)

        if salvarDisco:
            Utilidades.salvarEmDisco('imagem-inicial', 'jpeg', self.imagemValidada)
            Utilidades.salvarEmDisco('imagem-DCT', 'jpeg', imagemResDCT)
            Utilidades.salvarEmDisco('imagem-pixelEliminados', 'jpeg', imagemResEliminacao)
            Utilidades.salvarEmDisco('imagem-resultante', 'jpeg', imagemResultante)

        return imagemResultante


class DCTizador:
    def __init__(self, tamanhoBloco:int, blocosV:int, blocosH:int):
        self.tamanhoBloco = tamanhoBloco
        self.blocosV = blocosV
        self.blocosH = blocosH

    def aplicarDCT(self, imagem:np.ndarray):
        imagemTransformada = np.zeros((self.blocosV*self.tamanhoBloco,self.blocosH*self.tamanhoBloco))

        for linha in range(self.blocosV):
            for coluna in range(self.blocosH):
                    blocoAtual = cv.dct(imagem[linha*self.tamanhoBloco:(linha+1)*self.tamanhoBloco,coluna*self.tamanhoBloco:(coluna+1)*self.tamanhoBloco])
                    imagemTransformada[linha*self.tamanhoBloco:(linha+1)*self.tamanhoBloco,coluna*self.tamanhoBloco:(coluna+1)*self.tamanhoBloco]=blocoAtual
        # imagemTransformada = cv.dct(imagem)

        return imagemTransformada

    def aplicarIDCT(self, imagem:np.ndarray):
        imagemDestransformada = np.zeros((self.blocosV*self.tamanhoBloco,self.blocosH*self.tamanhoBloco))

        for linha in range(self.blocosV):
            for coluna in range(self.blocosH):
                    blocoAtual = cv.idct(imagem[linha*self.tamanhoBloco:(linha+1)*self.tamanhoBloco,coluna*self.tamanhoBloco:(coluna+1)*self.tamanhoBloco])
                    imagemDestransformada[linha*self.tamanhoBloco:(linha+1)*self.tamanhoBloco,coluna*self.tamanhoBloco:(coluna+1)*self.tamanhoBloco]=blocoAtual

        # imagemDestransformada = cv.idct(imagem)

        return imagemDestransformada

class PixelKiller:
    def __init__(self, tamanhoBloco:int, blocosV:int, blocosH:int):
        self.tamanhoBloco = tamanhoBloco
        self.blocosV = blocosV
        self.blocosH = blocosH

    def eliminarPixels(self, imagem:np.ndarray, fatorDeCompressao: int):
        limite = int(round(self.tamanhoBloco - ((fatorDeCompressao/100) * self.tamanhoBloco)))
        imagemLoss = np.zeros((self.blocosV*self.tamanhoBloco,self.blocosH*self.tamanhoBloco))

        for linha in range(self.blocosV):
            for coluna in range(self.blocosH):
                    blocoAtual = imagem[linha*self.tamanhoBloco:(linha+1)*self.tamanhoBloco,coluna*self.tamanhoBloco:(coluna+1)*self.tamanhoBloco]
                    blocoAtual[limite:self.tamanhoBloco, limite:self.tamanhoBloco] = 0
                    imagemLoss[linha*self.tamanhoBloco:(linha+1)*self.tamanhoBloco,coluna*self.tamanhoBloco:(coluna+1)*self.tamanhoBloco]=blocoAtual

        return imagemLoss

class Utilidades:
    
    @staticmethod
    def getMADEntreImagens(img1:np.ndarray, img2:np.ndarray, blocosV:int, blocosH:int):
        tamanhoBloco = 8
        diferenca = img1 - img2
        MAD = round(np.sum(np.abs(diferenca)) / float(blocosV*tamanhoBloco*blocosH*tamanhoBloco))
        print("Mean Absolute Difference: ",MAD)
        return MAD

    @staticmethod
    def salvarEmDisco(titulo:string, formato:string, img:np.ndarray):
        cv.imwrite(f"{titulo}.{formato}", img)
        return True
