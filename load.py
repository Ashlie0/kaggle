import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def loadim(a=-1,j="trainimages"):
    if j=="trainimages":
        img = os.listdir("./train/images/")
        place="./train/images/"
    elif j=="trainmasks":
        img = os.listdir("./train/masks/")
        place="./train/masks/"
    elif j=="test":
        img = os.listdir("./test/")
        place="./test/"
    img_in = []
    #開いたフォルダの中にある画像の名前をイテレーションで全て抽出する
    k=0
    for i in img:
        #if k>=120:
            #break
        #listdirで開くと画像の名前とは関係のないThumbs.dbが抽出されるので無視する
        if i == "Thumbs.db":
            continue
        #imagesの中にある全ての画像の配列を開いていく
        imagea = np.array(Image.open(place+i))
        #reshapeを使って開かれた配列を1次元配列に変換する
        imagea_resize = imagea.reshape(imagea.size)
        #バッチリストに追加していく
        img_in.append(imagea_resize)
        if k==a:
            print(i)
        k+=1

    if j=="trainmasks":
        x = np.array(img_in).reshape(k, 1, 101, 101)
    else:
        x = np.array(img_in).reshape(k, 3, 101, 101)#.astype(np.float32)
    if a >= 0:
        if j=="trainmasks":
            plt.imshow(x[a].reshape(101,101))
            plt.gray()
        else:
            plt.imshow(x[a].reshape(101,101,3))
        plt.show()
    plt.close()
    return x.astype(np.float32)/255 if j=="trainimages"or j=="test" \
        else x.astype(np.float32)/65535
def transans(t):
    t=t.reshape(101,101)
    k=[]
    j=False
    z=0
    for i in range(101*101):
        y=i//101
        x=i%101
        if t[x][y]==1:
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

ti=loadim(j="trainimages")
#tm=loadim(j="trainmasks")
#te=loadim(0,"test")


#print(transans(tm[1]))