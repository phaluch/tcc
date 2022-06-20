from tkinter import *
import random as rd
from tkinter import messagebox
from PIL import ImageTk, Image
import sys


class GameGrid(Frame):     #the game
    def __init__(self, master, height, width, mines_count, player):
        Frame.__init__(self, master)
        self.grid(row=0)
        self.master = master
        if sys.platform == 'win32':     #checking os
            self.platform = 'windows'
        else:
            self.platform = 'macos'
        self.height = height     #storing height, width, mines_count, and player's name
        self.width = width
        self.mines_count = mines_count
        self.player_name = player
        self.play_time = 0     #initiating play_time and other values
        self.lost = False
        self.won = False
        self.notmine = height * width - mines_count     #calculate the number of tiles that are not mines
        flag = Image.open('main/tkinter/ca.PNG')    #creating and storing flag and bomb images
        flag = flag.resize((12, 12), Image.ANTIALIAS)
        bomb = Image.open('main/tkinter/aurus.PNG')
        bomb = bomb.resize((12, 12), Image.ANTIALIAS)
        self.flag = ImageTk.PhotoImage(flag)
        self.bomb = ImageTk.PhotoImage(bomb)
        
        grid_model = [[0]*width for item in [0]*height]     #creating a list to hold 1's and 0's
        while mines_count > 0:                              #1 is mine, 0 is normal
            randi = rd.randint(0, height-1)     #putting mines into the list by generating random coordinates
            randj = rd.randint(0, width-1)      #and storing mine in the corresponding place
            if grid_model[randi][randj] == 0:
                grid_model[randi][randj] = 1
                mines_count -= 1
        self.tiles = {}     #creating Tiles and storing them using dictionary
        for i in range(height):
            for j in range(width):   
                if grid_model[i][j] == 1:
                    self.tiles[i, j] = Tile(self, i, j, True)
                else:       
                    mine_neighbors = 0     #counting nearby mines if Tile in creation is not a mine
                    if i - 1 >= 0:
                        if grid_model[i-1][j] == 1:
                            mine_neighbors += 1
                    if i - 1 >= 0 and j - 1 >= 0:
                        if grid_model[i-1][j-1] == 1:
                            mine_neighbors += 1
                    if i - 1 >= 0 and j + 1 < width:
                        if grid_model[i-1][j+1] == 1:
                            mine_neighbors += 1
                    if j - 1 >= 0:
                        if grid_model[i][j-1] == 1:
                            mine_neighbors += 1
                    if j + 1 < width:
                        if grid_model[i][j+1] == 1:
                            mine_neighbors += 1
                    if i + 1 < height:
                        if grid_model[i+1][j] == 1:
                            mine_neighbors += 1
                    if i + 1 < height and j - 1 >= 0:
                        if grid_model[i+1][j-1] == 1:
                            mine_neighbors += 1
                    if i + 1 < height and j + 1 < width:
                        if grid_model[i+1][j+1] == 1:
                            mine_neighbors += 1
                    
                    self.tiles[i, j] = Tile(self, i, j, False, mine_neighbors)
        
    def reveal_surroundings(self, i, j):     #reveal nearby tiles
        revealing = []
        width = self.width
        height = self.height

        
        if i - 1 >= 0:
            revealing.append(self.tiles[i-1, j])
        if i - 1 >= 0 and j - 1 >= 0:
            revealing.append(self.tiles[i-1, j-1])
        if i - 1 >= 0 and j + 1 < width:
            revealing.append(self.tiles[i-1, j+1])
        if j - 1 >= 0:
            revealing.append(self.tiles[i, j-1])
        if j + 1 < width:
            revealing.append(self.tiles[i, j+1])
        if i + 1 < height:
            revealing.append(self.tiles[i+1, j])
        if i + 1 < height and j - 1 >= 0:
            revealing.append(self.tiles[i+1, j-1])
        if i + 1 < height and j + 1 < width:
            revealing.append(self.tiles[i+1, j+1])


        for tile in revealing:
            tile.reveal()


    def lose(self):     #show if lost, stop the clock
        global stp
        stp = True
        self.lost = True
        if self.platform == 'windows':
            for tile in self.tiles:
                if self.tiles[tile].mine:
                    self.tiles[tile].config(bg='red')
        
        else:    
            for tile in self.tiles:
                if self.tiles[tile].mine:
                    self.tiles[tile].config(image=self.bomb, padx=9, pady=4, bg='red')
                self.tiles[tile].unbind('<Button-1>')
                self.tiles[tile].unbind('<Button-2>')   
        messagebox.showerror(message='Boom, Game Over!!')
        self.score = ScoreBoard(self.master)
        
        
    def win(self):      #show if won, stop the clock, creating a window recording scores
        global mn, sc, stp
        stp = True
        self.won = True
        for tile in self.tiles:
            if self.tiles[tile].mine:
                self.tiles[tile].config(image=self.bomb, padx=9, pady=4, bg='red')
            self.tiles[tile].unbind('<Button-1>')
            self.tiles[tile].unbind('<Button-2>')
        messagebox.showinfo(message='Congrats, You Survived ;)')
        
        play_time = str(m) + ' mins, ' + str(s) + ' secs'
        self.score = ScoreBoard(self.master, self.player_name, play_time)
                

                

class ScoreBoard(Toplevel):     #for score recording
    def __init__(self, master, name=None, time=None):
        Toplevel.__init__(self, master)
        self.title('Hall of Fame')                   
        fin_text = ''
        
        if name != None:    #writing in the record if there is one
            self.board = open('ScoreBoard.txt', 'r')    #assigning the text inside ScoreBoard.txt to board_text
            board_text = ''                             #and writing it into ScoreBoard.txt
            for line in self.board:
                board_text = board_text + line
            self.board = open('ScoreBoard.txt', 'w')
            self.record = name + ' ' + time 
            self.board.write(board_text + '\n' + self.record)
            
        self.board = open('ScoreBoard.txt', 'r') #reading text in ScoreBoard and put it on the window

        for line in self.board:
            fin_text = fin_text + line
        self.lbl = Label(self, text=fin_text)
        self.lbl.pack()
        self.geometry('300x300')
        self.board.close()
            
            
            
        
    
class Tile(Label):      #the Tile
    def __init__(self, master, i, j, mine, mine_neighbors=None):
        Label.__init__(self, master, width=2, relief=RAISED)
        self.grid(row=i, column=j)
        self.game = master     #storing row, column, is mine or not, count of nearby mines
        self.mine = mine
        self.row = i
        self.col = j
        self.mine_neighbors = mine_neighbors
        self.revealed = False
        self.marked = False
        self.bind('<Button-1>', self.reveal)    #bind Tile: reveal(left click), mark(right click)
        self.bind('<Button-2>', self.mark)

    def reveal(self, event=None):       #revealing tile
        if self.mine:
            self.game.lose()
            return
        else:
            if not self.revealed:
                self.revealed = True
                self.mark()
                self.unbind('<Button-1>')
                self.unbind('<Button-2>')
                if self.mine_neighbors == 0:    #if no nearby mines, reveal nearby tiles
                    self.config(text='', relief=SUNKEN, bg='lightgrey', image='', padx=1, pady=1)
                    self.game.reveal_surroundings(self.row, self.col)
                else:
                    self.config(text=self.mine_neighbors, relief=SUNKEN, bg='lightgrey', image='', padx=1, pady=1)
                self.game.notmine -= 1

            
            if self.game.notmine == 0:
                self.game.win() 

    def mark(self,event=None):      #marking tile
        if self.game.platform == 'windows':
            if not self.marked:
                self.config(text='*')
                self.marked = True
            else:
                self.config(text='')
                self.marked = False
    
        else:
            if not self.marked:
                self.config(image=self.game.flag, padx=9, pady=4)
                self.marked = True
            else:
                self.config(image='', padx=1, pady=1)
                self.marked = False

                
stp = False     #used to stop the clock when lost or won


def update_time():     #a stopwatch
    global m, s, timer, stp
    if stp != True:
        s = s + 1
        if s == 60:
            m = m + 1
            s = 0

        mn = str(m)     #making the clock look better by adding a 0 when the number
        sc = str(s)     #of second or minute is just one digit, e.g. 01, 06, 09..
        if len(sc) == 1 and len(mn) == 1:
            sc = '0' + sc
            mn = '0' + mn
            timer.config(text=mn+':'+sc)
        elif len(mn) == 1 and len(sc) != 1:
            mn = '0' + str(m)
            timer.config(text=mn+':'+str(s))
        elif len(sc) == 1 and len(mn) != 1:
            sc = '0' + sc
            timer.config(text=mn+':'+sc)
    
        timer.after(1000, update_time)


def play(height, width, mines_count, player):       #initiating the game
    global s, m, timer
    m = 0
    s = -1

    time = str(m) + ':' + str(s)
    
    root = Tk()
    root.title('MineSweeper')
    root.resizable(False, False)

    timer = Label(root, text='%i:%i'%(m,s))     #creating stopwatch and update it every second
    timer.grid(row=1)
    update_time()
    
    game = GameGrid(root, height, width, mines_count, player)
    root.mainloop()
    
if __name__ == '__main__':   
    play(10, 10, 10, 'Harley')