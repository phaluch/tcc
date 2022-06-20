from distutils.command.build import build
import cv2 as cv
import tkinter as tk
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
import numpy as np

W, H = 800, 400
#img = cv.imread('main/tkinter/8x8.png',cv.IMREAD_GRAYSCALE)
img = cv.imread('main/tkinter/ca.PNG',cv.IMREAD_GRAYSCALE)
#print(img)
i25 = cv.imread('main/tkinter/25x25.png',cv.IMREAD_GRAYSCALE)
#i16 = cv.imread('16x16.png',cv.IMREAD_GRAYSCALE)
#img = cv.resize(img, (int(W/2), H), interpolation=cv.INTER_NEAREST)
root = tk.Tk()
canvas = tk.Canvas(root, width=W, height=H)


def coef(k_or_j,m_or_n):
    if k_or_j == 0:
        return np.sqrt(1/m_or_n)
    return np.sqrt(2/m_or_n)

def C(m,n,k,j):
    M = np.zeros((m,n))
    for r in range(m):
        for s in range(n):
            term1 = coef(k,m)*coef(j,n)
            term2 = np.cos(np.pi*k*(r+1/2)/m)
            term3 = np.cos(np.pi*j*(s+1/2)/n)
            M[r,s] = term1*term2*term3
    return M

def pixelateThis(img, width, height, pw=8, ph=8):
    # Resize img to "pixelated" size
    temp = cv.resize(img, (pw, ph), interpolation=cv.INTER_LINEAR)

    # Initialize output image
    output = cv.resize(temp, (width, height), interpolation=cv.INTER_NEAREST)
    return output

def resizeThis(img, w=16, h=16):
    #print(img)
    # Initialize output image
    output = cv.resize(img, (w, h), interpolation=cv.INTER_NEAREST)
    return output


def labelFromArray(arr, column, row, size):

    # Imagem
    logo = resizeThis(arr, *size)

    logo = Image.fromarray(logo)

    logo = ImageTk.PhotoImage(logo)

    logo_label = tk.Label(image=logo)
    logo_label.image = logo

    # Placing image
    logo_label.grid(column=column,row=row)

#
def c():
    global flag
    if flag == 1:
        # Imagem
        logo = Image.open('main/tkinter/ca.PNG')
        logo = ImageTk.PhotoImage(logo)
        flag=0
    else:
        # Imagem
        logo = Image.open('main/tkinter/aurus.PNG')
        logo = ImageTk.PhotoImage(logo)
        flag=1
        

def buildImage():
    global labels, m, n
    global mask
    global img
    global imgdct
    global packed_arr
    global logo_label
    out = np.zeros((m, n))

    k,j = 0,0

    print(len(packed_arr))
    print('shape: ',packed_arr[0][0].shape)
    for row_M, row_dct in zip(packed_arr,imgdct):
        for M, dct in zip(row_M, row_dct):
            #print(f'build: Testing mask[{k}][{j}]')
            #print(f'build: Testing mask[{k}][{j}]: {mask[k][j]}')
            if mask[j][k] == 1:
                print('out',out.shape)
                print('M', M.shape)
                print(imgdct[k][j])
                out = out + M*imgdct[k][j]
            j+=1
        k+=1
        j=0
    out = resizeThis(out, int(W/2), H)
    black = Image.fromarray(out)
    black = ImageTk.PhotoImage(black)
    logo_label.configure(image=black)
    logo_label.image=black


# Botao
'''browse_text = tk.StringVar()
browse_btn = tk.Button(root, textvariable=browse_text, command=lambda:c(), bg="#20bebe")
browse_text.set("Muda!")
browse_btn.grid(column=2)
'''

m = 8
n = 8

canvas.grid(columnspan=m, rowspan=n)
labels = [[None for j in range(n)] for i in range(m)]

for i in range(m):
    for j in range(n):
        labels[i][j] = tk.Button(root, text=f'[{i}][{j}]', command= lambda j=i, k=j: changeButton(k,j))

mask = [[1 for j in range(n)] for i in range(m)]

# Imagem
logo = pixelateThis(img, int(W/2), H, m, n)
actual_logo = resizeThis(img, m, n)
imgdct = cv.dct(actual_logo.astype(float))
print(imgdct.shape)
print(imgdct[0,0])
logo = Image.fromarray(logo)

logo = ImageTk.PhotoImage(logo)

logo_label = tk.Label(image=logo)
logo_label.image = logo

def changeButton(j,k):
    global mask, m, n, packed_sequence
    print(f'Testing mask[{k}][{j}]: {mask[j][k]}')
    if mask[j][k] == 0:
        labels[j][k].configure(image=packed_sequence[j][k])
        labels[j][k].image = packed_sequence[j][k]
        mask[j][k] = 1
    else:
        black = Image.fromarray(np.zeros((m,n))+255)
        black = ImageTk.PhotoImage(black)
        labels[j][k].configure(image=black)
        labels[j][k].image = black
        mask[j][k] = 0
    
    buildImage()

# Placing image
logo_label.grid(column=1, row=1, columnspan=int(m/2), rowspan=n)
packed_sequence = []
packed_arr = []
sequence = []
for j in range(m):
    packed_row = []
    packed_row_arr = []
    for k in range(n):
        current_block = C(m,n,k,j)
        #print(current_block)
        cur_delta = current_block - 1/128
        #print(cur_delta)
        #print(a[k,j])
        to_append = current_block*imgdct[k,j]
        current_block /= np.max(np.abs(current_block),axis=0)
        current_block *= (255.0/current_block.max())
        #print(to_append)
        #sequence.append(to_append*255)
        packed_row_arr.append(to_append*255)
        resized = resizeThis(current_block, int(W/(2*m)),int(H/n))
        img = Image.fromarray(resized)
        img = ImageTk.PhotoImage(img)
        packed_row.append(img)
        labels[k][j].configure(image=img)
        labels[k][j].image = img
        labels[k][j].grid(column=int(W/m)+k+1,row=j+1)
    packed_sequence.append(packed_row)
    packed_arr.append(packed_row_arr)

'''for ax, M in tqdm(zip(axs.flat, sequence)):
    ax.imshow(M, cmap='gray')
plt.tight_layout()
plt.savefig('t1p.png')'''

#Resolvendo unever gaps




root.mainloop()