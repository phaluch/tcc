import string
from xmlrpc.client import boolean
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm


class Compressor:
    tamanhoBloco = 8

    def __init__(self, imagemOriginal: np.ndarray):
        self.imagemOriginal = imagemOriginal
        self.h, self.w = (np.array(self.imagemOriginal.shape[:2])/self.tamanhoBloco * self.tamanhoBloco).astype(int)
        self.blocosV = int(self.h/self.tamanhoBloco)
        self.blocosH = int(self.w/self.tamanhoBloco)
        self.imagemValidada = np.zeros((self.blocosV*self.tamanhoBloco,self.blocosH*self.tamanhoBloco))
        self.imagemValidada[:self.blocosV*self.tamanhoBloco,:self.blocosH*self.tamanhoBloco] = self.imagemOriginal[:self.blocosV*self.tamanhoBloco,:self.blocosH*self.tamanhoBloco]

    def comprimir(self, salvarDisco:bool = False):
        dctizador = DCTizador(self.tamanhoBloco, self.blocosV, self.blocosH)
        quantizador = Quantizador(self.tamanhoBloco, self.blocosV, self.blocosH)
        imagemResDCT = dctizador.aplicarDCT(self.imagemValidada)
        # imagemResQuantizacao = quantizador.quantizarImagem(imagemResDCT)
        # imagemResDesquantizacao = quantizador.dequantizarImagem(imagemResQuantizacao)
        imagemResDesquantizacao = quantizador.eliminarDados(imagemResDCT, 80, self.h, self.w)
        imagemResultante = dctizador.aplicarIDCT(imagemResDesquantizacao)

        #Utilidades.getMADEntreImagens(self.imagemValidada, imagemResultante, self.blocosV, self.blocosH)

        if salvarDisco:
            Utilidades.salvarEmDisco('imagem-inicial', 'jpeg', self.imagemValidada)
            Utilidades.salvarEmDisco('imagem-DCT', 'jpeg', imagemResDCT)
            # Utilidades.salvarEmDisco('imagem-quantizada', 'jpeg', imagemResQuantizacao)
            Utilidades.salvarEmDisco('imagem-resultante', 'jpeg', imagemResultante)

        return imagemResultante

        


class DCTizador:
    def __init__(self, tamanhoBloco:int, blocosV:int, blocosH:int):
        self.tamanhoBloco = tamanhoBloco
        self.blocosV = blocosV
        self.blocosH = blocosH

    def aplicarDCT(self, imagem:np.ndarray):
        # imagemTransformada = np.zeros((self.blocosV*self.tamanhoBloco,self.blocosH*self.tamanhoBloco))

        # for linha in range(self.blocosV):
        #     for coluna in range(self.blocosH):
        #             blocoAtual = cv.dct(imagem[linha*self.tamanhoBloco:(linha+1)*self.tamanhoBloco,coluna*self.tamanhoBloco:(coluna+1)*self.tamanhoBloco])
        #             imagemTransformada[linha*self.tamanhoBloco:(linha+1)*self.tamanhoBloco,coluna*self.tamanhoBloco:(coluna+1)*self.tamanhoBloco]=blocoAtual
        imagemTransformada = cv.dct(imagem)

        return imagemTransformada

    def aplicarIDCT(self, imagem:np.ndarray):
        # imagemDestransformada = np.zeros((self.blocosV*self.tamanhoBloco,self.blocosH*self.tamanhoBloco))

        # for linha in range(self.blocosV):
        #     for coluna in range(self.blocosH):
        #             blocoAtual = cv.idct(imagem[linha*self.tamanhoBloco:(linha+1)*self.tamanhoBloco,coluna*self.tamanhoBloco:(coluna+1)*self.tamanhoBloco])
        #             imagemDestransformada[linha*self.tamanhoBloco:(linha+1)*self.tamanhoBloco,coluna*self.tamanhoBloco:(coluna+1)*self.tamanhoBloco]=blocoAtual

        imagemDestransformada = cv.idct(imagem)

        return imagemDestransformada

class Quantizador:
    matrizQuantizacao = [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 61, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ]

    def __init__(self, tamanhoBloco:int, blocosV:int, blocosH:int):
        self.tamanhoBloco = tamanhoBloco
        self.blocosV = blocosV
        self.blocosH = blocosH

    def quantizarBloco(self, bloco:np.ndarray):
        for i in range(self.tamanhoBloco):
            for j in range(self.tamanhoBloco):
                bloco[i][j] = int(round(bloco[i][j] / self.matrizQuantizacao[i][j]))
        return bloco

    def dequantizarBloco (self, bloco:np.ndarray):
        for i in range(self.tamanhoBloco):
            for j in range(self.tamanhoBloco):
                bloco[i][j] = int(round(bloco[i][j] * self.matrizQuantizacao[i][j]))
        return bloco

    def quantizarImagem(self, imagem:np.ndarray):
        imagemQuantizada = np.zeros((self.blocosV*self.tamanhoBloco,self.blocosH*self.tamanhoBloco))

        for linha in range(self.blocosV):
            for coluna in range(self.blocosH):
                    blocoAtual = self.quantizarBloco(imagem[linha*self.tamanhoBloco:(linha+1)*self.tamanhoBloco,coluna*self.tamanhoBloco:(coluna+1)*self.tamanhoBloco])
                    imagemQuantizada[linha*self.tamanhoBloco:(linha+1)*self.tamanhoBloco,coluna*self.tamanhoBloco:(coluna+1)*self.tamanhoBloco]=blocoAtual

        return imagemQuantizada

    def eliminarDados(self, imagem:np.ndarray, fatorDeCompressao: int, h, w):
        lh = int(round(h - ((fatorDeCompressao/100) * h)))
        lw = int(round(w - ((fatorDeCompressao/100) * w)))

        imagemRecebida = imagem
        imagemRecebida[lh:h, lw:w] = 0
        return imagemRecebida

    def dequantizarImagem(self, qImagem: np.ndarray):
        imagemDequantizada = np.zeros((self.blocosV*self.tamanhoBloco,self.blocosH*self.tamanhoBloco))

        for linha in range(self.blocosV):
            for coluna in range(self.blocosH):
                    blocoAtual = self.dequantizarBloco(qImagem[linha*self.tamanhoBloco:(linha+1)*self.tamanhoBloco,coluna*self.tamanhoBloco:(coluna+1)*self.tamanhoBloco])
                    imagemDequantizada[linha*self.tamanhoBloco:(linha+1)*self.tamanhoBloco,coluna*self.tamanhoBloco:(coluna+1)*self.tamanhoBloco]=blocoAtual

        return imagemDequantizada

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
