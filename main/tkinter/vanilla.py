from distutils.command.build import build
import cv2 as cv
import tkinter as tk
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
import numpy as np
from random import randint
from tqdm import tqdm

def coef(k_or_j,m_or_n):
    if k_or_j == 0:
        return np.sqrt(1/m_or_n)
    return np.sqrt(2/m_or_n)

def C(m,n,i,j):
    M = np.zeros((m,n))
    for r in range(m):
        for s in range(n):
            term1 = coef(i,m)*coef(j,n)
            term2 = np.cos(np.pi*i*(r+1/2)/m)
            term3 = np.cos(np.pi*j*(s+1/2)/n)
            M[r,s] = term1*term2*term3
    return M

def prepImage(img, width=None, height=None):
    if type(img) == str:
        img = cv.imread(img)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    return img

def resizeThis(img, w=16, h=16):
    #print(img)
    # Initialize output image
    output = cv.resize(img, (w, h), interpolation=cv.INTER_NEAREST)
    return output

def pixelateThis(img, width, height, pw=8, ph=8):
    # Resize img to "pixelated" size
    temp = cv.resize(img, (pw, ph), interpolation=cv.INTER_LINEAR)

    # Initialize output image
    output = cv.resize(temp, (width, height), interpolation=cv.INTER_NEAREST)
    return output


W, H = 1600, 800
#img = cv.imread('main/tkinter/8x8.png',cv.IMREAD_GRAYSCALE)
i8_raw = cv.imread('main/tkinter/dratini.png',cv.IMREAD_GRAYSCALE)
#print(img)
i25 = cv.imread('main/tkinter/25x25.png',cv.IMREAD_GRAYSCALE)
#i16 = cv.imread('16x16.png',cv.IMREAD_GRAYSCALE)
#img = cv.resize(img, (int(W/2), H), interpolation=cv.INTER_NEAREST)
root = tk.Tk()
canvas = tk.Canvas(root, width=W, height=H)
#
def c():
    global flag
    if flag == 1:
        # Imagem
        logo = Image.open('main/tkinter/aurus.PNG')
        logo = ImageTk.PhotoImage(logo)
        flag=0
    else:
        # Imagem
        logo = Image.open('main/tkinter/aurus.PNG')
        logo = ImageTk.PhotoImage(logo)
        flag=1

def NormalizeData(data, range=255):
    return range * (data - np.min(data)) / (np.max(data) - np.min(data))
        

# Botao
'''browse_text = tk.StringVar()
browse_btn = tk.Button(root, textvariable=browse_text, command=lambda:c(), bg="#20bebe")
browse_text.set("Muda!")
browse_btn.grid(column=2)
'''


m = 40
n = 40

i8 = cv.resize(i8_raw, (m, n), interpolation=cv.INTER_LINEAR)

i8dct = cv.dct(i8.astype(float))
print('i8dct.shape: ',i8dct.shape)

padding_cols = 4

canvas.grid(columnspan=n+padding_cols, rowspan=m)
labels = [[None for j in range(n)] for i in range(m)]
flags = np.array([[None for j in range(n)] for i in range(m)])
bases = []
for i in tqdm(range(m)):
    bases_row = []
    for j in range(n):
        freq_base = C(m,n,i,j)
        bases_row.append(freq_base)
    bases.append(bases_row)

def buildImage():
    global i8dct, m, n
    cur_base = np.zeros((m, n))
    for a in range(m):
        for b in range(n):
            if flags[a][b]:
                #print(f'+flags[{a}][{b}]')
                cur_base+=bases[a][b]*i8dct[a][b]
                #print('*',i8dct[a][b])
    try:
        normalized = NormalizeData(cur_base)
        return normalized
    except:
        print(cur_base)
        print('--------------------')
        raise Exception

instrucao_X = tk.Label(root, text='Intervalo X:')
instrucao_X.grid(row=int(n/2)+5, column=0)    
valor_de_X = tk.Entry(root)
valor_de_X.grid(row=int(n/2)+5, column=1)

instrucao_Y = tk.Label(root, text='Intervalo Y:')
instrucao_Y.grid(row=int(n/2)+5, column=2)    
valor_de_Y = tk.Entry(root)
valor_de_Y.grid(row=int(n/2)+5, column=3)


def pegarValores():
    global valor_de_X, valor_de_Y
    x = valor_de_X.get()
    y = valor_de_Y.get()
    if not y:
        y='::'
    if not x:
        x='::'
    x = [int(i) for i in x.split(':')]
    y = [int(j) for j in y.split(':')]
    return y, x

def adicionar():
    global flags
    x, y = pegarValores()
    flags[x[0]:x[1]:x[2],y[0]:y[1]:y[2]] = 1
    normalized = buildImage()
    img = Image.fromarray(pixelateThis(normalized,400,400,m,n))
    img = ImageTk.PhotoImage(img)
    but1.configure(image=img)
    but1.image=img
    updateButtons()
 

def subtrair():
    global flags
    x, y = pegarValores()
    flags[x[0]:x[1]:x[2],y[0]:y[1]:y[2]] = 0
    normalized = buildImage()
    img = Image.fromarray(pixelateThis(normalized,400,400,m,n))
    img = ImageTk.PhotoImage(img)
    but1.configure(image=img)
    but1.image=img
    updateButtons()

def updateButtons():
    global but1, flags, m, n, labels, bases
    for i in range(m):
        for j in range(n):
            if not flags[i][j]:
                filler = np.zeros((m, n))
                img = Image.fromarray(pixelateThis(filler,botao_frequencia_w, botao_frequencia_h,m,n))
                img = ImageTk.PhotoImage(img)
                labels[i][j].configure(image=img)
                labels[i][j].image = img
            else:
                freq_base = abs(NormalizeData(pixelateThis(bases[i][j],botao_frequencia_w, botao_frequencia_h,m,n)))
                #print(freq_base)
                img = prepImage(freq_base)
                labels[i][j].configure(image=img)
                labels[i][j].image = img

adicionarButton = tk.Button(root, text='Adicionar', command = lambda:adicionar())
adicionarButton.grid(row=int(n/2)+6, column=1)
subtrairButton = tk.Button(root, text='Subtrair', command = lambda:subtrair())
subtrairButton.grid(row=int(n/2)+6, column=2)


botao_frequencia_w, botao_frequencia_h = 20, 20

for i in range(m):
    for j in range(n):
        freq_base = abs(NormalizeData(pixelateThis(bases[i][j],botao_frequencia_w, botao_frequencia_h,m,n)))
        #print(freq_base)
        img = prepImage(freq_base)
        labels[i][j] = tk.Button(root, text=f'[{i}][{j}]', command = lambda i=i, j=j:chbutton3(i,j))
        labels[i][j].configure(image=img)
        labels[i][j].image=img
        labels[i][j].grid(row=i,column=j+padding_cols)
        flags[i][j] = True


base = np.zeros((m, n))

f1 = base.copy()
f1[:,::2] = 300

f2 = base.copy()
f2[::3,::4] = 50

def chbutton():
    global but1
    n = randint(0,2)
    if n==0:
        arr = resizeThis(f1+f2,400,400)
        img = Image.fromarray(arr)
    elif n==1:
        arr = resizeThis(f1,400,400)
        img = Image.fromarray(arr)
    elif n==2:
        arr = resizeThis(f2,400,400)
        img = Image.fromarray(arr)
    img = ImageTk.PhotoImage(img)
    but1.configure(image=img)
    but1.image=img
    print('n: ',n)
        

def chbutton2(i,j):
    global but1, m, n, labels, flags
    cur_base = np.zeros((m, n))
    if flags[i][j]:
        flags[i][j] = False
    else:
        flags[i][j] = True
    for a in range(m):
        for b in range(n):
            if flags[a][b]:
                #print(f'+flags[{a}][{b}]')
                cur_base+=bases[a][b]
    normalized = NormalizeData(cur_base)
    img = Image.fromarray(resizeThis(normalized,400,400))
    img = ImageTk.PhotoImage(img)
    but1.configure(image=img)
    but1.image=img

def chbutton3(i,j):
    global but1, flags, m, n, labels, bases
    if flags[i][j]:
        flags[i][j] = False
        filler = np.zeros((m, n))
        img = Image.fromarray(pixelateThis(filler,botao_frequencia_w, botao_frequencia_h,m,n))
        img = ImageTk.PhotoImage(img)
        labels[i][j].configure(image=img)
        labels[i][j].image = img
    else:
        flags[i][j] = True
        freq_base = abs(NormalizeData(pixelateThis(bases[i][j],botao_frequencia_w, botao_frequencia_h,m,n)))
        #print(freq_base)
        img = prepImage(freq_base)
        labels[i][j].configure(image=img)
        labels[i][j].image = img
    normalized = buildImage()
    img = Image.fromarray(pixelateThis(normalized,400,400,m,n))
    img = ImageTk.PhotoImage(img)
    but1.configure(image=img)
    but1.image=img

but1 = tk.Button(root, text=f'BIGONE')
arr = pixelateThis(i8,400,400,m,n)
img = Image.fromarray(arr)
img = ImageTk.PhotoImage(img)
but1.configure(image=img, command = lambda:chbutton() )
but1.image = img
but1.grid(row=0, column=0, rowspan=int(n/2), columnspan=padding_cols)




root.mainloop()