import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import imutils
import glob
import pandas as pd


def pixel_per_cm2(area_lar, area_alt):
    return (dist_in_pixel_larg * dist_in_pixel_alt) / (area_alt * area_lar)

medicoes=[]
areat = []
parcelas=[]
i=1

os.chdir('D:\My Drive\Arquivos-notebook\Python-projects\Eucalipto-rachadura\python')
#######CARREGAR IMAGEM
path_of_images='D:\My Drive\Arquivos-notebook\Python-projects\Eucalipto-rachadura\python\Test'####direterio das imagens
outfile='D:\My Drive\Arquivos-notebook\Python-projects\Eucalipto-rachadura\python\Referencia'
saida='D:\My Drive\Arquivos-notebook\Python-projects\Eucalipto-rachadura\python\Rachadura'
saida_tora = 'D:\My Drive\Arquivos-notebook\Python-projects\Eucalipto-rachadura\python\Face'
filenames= glob.glob(path_of_images + "/*.jpg")


for imagem in filenames:

    img = cv2.imread(imagem)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.inRange(gray, 180, 255)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Sort contours from left to right as leftmost contour is reference object
    (cnts, _) = contours.sort_contours(cnts)

    # Remove contours which are not large enough
    cnts = [x for x in cnts if cv2.contourArea(x) > 1000]
    contorno = cv2.drawContours(img, cnts, 0, (0, 0, 255), 15)
    nome_legenda = os.path.basename(imagem)
    cv2.imwrite(os.path.join(outfile, nome_legenda), contorno)

    #######ENCONTRAR A REFÊRENCIA

    ref_object = cnts[0]
    box = cv2.minAreaRect(ref_object)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="float")
    box = perspective.order_points(box)
    (tl, tr, br, bl) = box
    dist_in_pixel_larg = euclidean(tl, tr)
    dist_in_pixel_alt = euclidean(bl, tl)
    pixel_cm2 = pixel_per_cm2(3, 3)
    Um_pixel_cm2 = (1 / pixel_cm2)

    #######ENCONTRAR RACHADURA
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hedged = cv2.GaussianBlur(hsv, (7, 7), 0)
    hedged = cv2.dilate(hedged, None, iterations=3)
    hedged = cv2.erode(hedged, None, iterations=3)
    h, s, v = cv2.split(hedged)

    rachadura = cv2.inRange(v, 0, 50)
    ntest = cv2.countNonZero(rachadura)
    md = (ntest * Um_pixel_cm2).__round__(4)
    plt.imshow(rachadura)
    cv2.imwrite(os.path.join(saida, nome_legenda), rachadura)  ####Foto rachadura

    ### ENCONTRAR A ÁREA DA TORA
    l = np.array([0, 0, 0])
    u = np.array([255, 210, 200])
    tora = cv2.inRange(hedged, l, u)
    cnts = cv2.findContours(tora, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # Remove contours which are not large enough
    cnts = [x for x in cnts if cv2.contourArea(x) > 1000]
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask = cv2.drawContours(mask.copy(), [max(cnts, key=cv2.contourArea)], -1, (255,255,255), -1)
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    areatora = cv2.countNonZero(mask)
    atora = (areatora * Um_pixel_cm2).__round__(4)
    cv2.imwrite(os.path.join(saida_tora, nome_legenda), mask)  ####Foto face

    ##########Salvar itens
    medicoes.append(md)
    parcelas.append(nome_legenda)
    areat.append(atora)
    a=(i/(len(filenames))*100)
    print("{0:.2f} % completed".format(round(a, 2)))
    i=i+1
tabela= np.array([parcelas,medicoes,areat])
tabela=np.transpose(tabela)
df = pd.DataFrame (tabela)
df.columns = ['Foto','Rachadura(cm2)','Area_Face(cm2)']
df.to_csv("medidas.csv",index=False,)

