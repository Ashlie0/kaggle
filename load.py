import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import csv
####################################################################
#def loadcsv():
#    with open("train.csv", "r") as f:
#        reader=csv.reader(f)
#        next(reader)
#        return [t for t in reader]
#def MaskToImage(t):
#    re=np.zeros([101,101])
#    for i in range(0,len(t),2):
#        for j in range(t[i],t[i]+t[i+1]):
#            y=j//101
#            x=j%101
#            re[x][y]=1
#    return re

#x=loadcsv()
#x=sorted(x, key=lambda t:t[0])
#y=[list(map(int,(t[1].split()))) for t in x]
#x=[t[0] for t in x]

#trainim=MaskToImage(y[1])
#showgray(trainim)
####################################################################
def showgray(t):
    plt.imshow(t.reshape(101,101))
    plt.show()
    plt.close()
def showrgb(t):
    plt.imshow(t.reshape(101,101,3))
    plt.show()
    plt.close()

def loadim(a=-1,j="trainimages"):
    if j=="trainimages":
        img = os.listdir("./traingray/")
        place="./traingray/"
    elif j=="trainmasks":
        img = os.listdir("./train/masks/")
        place="./train/masks/"
    elif j=="test":
        img = os.listdir("./test/")
        place="./test/"
    img_in = []
    #開いたフォルダの中にある画像の名前をイテレーションで全て抽出する
    name=[]
    k=0
    for i in img:
        #if k>=120:
        #    break
        #listdirで開くと画像の名前とは関係のないThumbs.dbが抽出されるので無視する
        if i == "Thumbs.db":
            continue
        name.append(i)
        #imagesの中にある全ての画像の配列を開いていく
        imagea = np.array(Image.open(place+i))
        #reshapeを使って開かれた配列を1次元配列に変換する
        imagea_resize = imagea.reshape(imagea.size)
        #バッチリストに追加していく
        img_in.append(imagea_resize)
        if k==a:
            print(i)
        k+=1

    x=np.array(img_in).reshape(k, 1, 101, 101)
    if a >= 0:
        showgray(x[a])
    x=x.astype(np.float32)
    if j=="trainimages"or j=="test":
        x/=255
    else:
        x/=65535
    return x
def loadcsv():
    with open("depths.csv", "r") as f:
        reader=csv.reader(f)
        next(reader)
        re=[t for t in reader]
        return re[0:4000:],re[4001::]
def ImageToMask(t):
    t=t.reshape(101,101)
    k=[]
    j=False
    z=0
    for i in range(101*101):
        y=i//101
        x=i%101
        if t[x][y]!=0:
            z+=1
            if j:
                continue
            k.append(str(i))
            j=True
        elif j:
            j=False
            k.append(str(z))
            z=0
    return " ".join(k)

    


######################################################################

tr_im=loadim(a=-1,j="trainimages")
tr_ma=loadim(a=-1,j="trainmasks")
teimage=loadim(a=-1,j="test")

trdepth,tedepth=loadcsv()
trdepth=sorted(trdepth,key=lambda t:t[0])
tedepth=sorted(tedepth,key=lambda t:t[0])

tr_dep=[int(t[1]) for t in trdepth]
te_dep=[int(t[1]) for t in tedepth]

tr_im_ma_dep=[[tr_im[i],tr_ma[i],tr_dep[i]] for i in range(4000)]
tr_im_ma_dep=sorted(tr_im_ma_dep,key=lambda t:t[2])


ma_dep_research=[0 for _ in range(20)]
count=[0 for _ in range(20)]
for t in tr_im_ma_dep:
    ma_dep_research[t[2]//50]+=np.sum(t[1])
    count[t[2]//50]+=1
x=[ma_dep_research[i]//max(1,count[i]) for i in range(20)]
print(x)








