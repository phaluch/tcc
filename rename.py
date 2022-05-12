import os

path = 'vinframes/'

arquivos = [x for x in os.listdir(path) if x.endswith('.jpeg')]
print(arquivos[:10])

for arquivo in arquivos:
    novo  = arquivo.replace('p.jpeg','.jpeg').replace('i.jpeg','.jpeg')
    os.rename(path+arquivo,path+novo)
