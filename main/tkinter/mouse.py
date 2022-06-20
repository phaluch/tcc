from tkinter import *
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from random import randint
import cv2 as cv

root = Tk()

images = {0: ImageTk.PhotoImage(Image.fromarray(np.zeros((30,30))+10)),
          1: ImageTk.PhotoImage(Image.fromarray(np.zeros((30,30))+100)),
          2: ImageTk.PhotoImage(Image.fromarray(np.zeros((30,30))+200))}

fill = 1

def pixelateThis(img, width, height, pw=8, ph=8):
    # Resize img to "pixelated" size
    temp = cv.resize(img, (pw, ph), interpolation=cv.INTER_LINEAR)

    # Initialize output image
    output = cv.resize(temp, (width, height), interpolation=cv.INTER_NEAREST)
    return output


def prepImage(img, width=None, height=None):
    if type(img) == str:
        img = cv.imread(img)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    return img


def NormalizeData(data, range=255):
    return range * (data - np.min(data)) / (np.max(data) - np.min(data))
    

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


class MyButton(Button): #Convention is for class names to start with uppercase letters
    def __init__(self, master, image):
        super(MyButton, self).__init__(master, borderwidth = 0)
        self.type = 0
        self.already_changed = False
        self.config(image=image)


    def change(self):
        if self.type == fill:
            self.type = 0
        else:
            self.type = fill
        self.config(image=images[self.type])

    def mouse_entered(self):
        if not self.already_changed:
            self.change()
            self.already_changed = True

    def mouse_up(self):
        self.already_changed = False

class Container(Frame):
    def __init__(self, master, width, height):
        super(Container, self).__init__(master)

                
        bases = []
        for i in range(width):
            bases_row = []
            for j in range(height):
                freq_base = C(width,height,i,j)
                bases_row.append(freq_base)
            bases.append(bases_row)


        buttons = []

        for y in range(height):
            buttons.append([])
            for x in range(width):
                freq_base = abs(NormalizeData(pixelateThis(bases[x][y],width, height,width, height)))
                #print(freq_base)
                img = prepImage(freq_base)
                button = MyButton(self,img)
                button.grid(row = x, column = y)
                buttons[y].append(button)

        self.buttons = buttons

        self.bind_all("<Button-1>", self.mouse_down)
        self.bind_all("<ButtonRelease-1>", self.mouse_up)
        self.bind_all("<B1-Motion>", self.mouse_motion)

        self.mouse_pressed = False

    def mouse_down(self, e):
        self.update_containing_button(e)
        self.mouse_pressed = True

    def mouse_up(self, e):
        self.mouse_pressed = False
        for row in self.buttons:
            for button in row:
                button.mouse_up()

    def mouse_motion(self, e):
        self.update_containing_button(e)

    def update_containing_button(self, e):
        for row in self.buttons:
            for button in row:
                if self.winfo_containing(e.x_root, e.y_root) is button:
                    button.mouse_entered()

grid = Container(root, 15, 15)
grid.pack()

root.mainloop()