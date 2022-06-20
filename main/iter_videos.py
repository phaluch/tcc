import os

for v in ['video.mp4','video2.mp4','video3.mp4']:
    os.system(f'python fluxo.py "{v}"')




def cut(x,p):
    x[:,int(1920*(1-p)):] = 0
    x[int(1080*(1-p)):,:] = 0
    return x


def daslice(x):
    return x[75:350,700:1100]