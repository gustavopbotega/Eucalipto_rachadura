import cv2
import numpy as np
import os
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import imutils
import glob
import pandas as pd
import matplotlib.pyplot as plt

def pixel_per_cm2(dist_in_pixel_larg,dist_in_pixel_alt,area_lar, area_alt):
    return (dist_in_pixel_larg * dist_in_pixel_alt) / (area_alt * area_lar)

def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

medicoes=[]
areat = []
parcelas=[]
proc =1

os.chdir('D:\My Drive\Projetos\Rachadura_Eucalipto\Analise rachadura')
#######CARREGAR IMAGEM
path_of_images='D:\My Drive\Projetos\Rachadura_Eucalipto\Analise rachadura\Fotos'####direterio das imagens
outfile='D:\My Drive\Python-projects\Eucalipto-rachadura\python\W_Outro metodo\Ref'
saida='D:\My Drive\Python-projects\Eucalipto-rachadura\python\W_Outro metodo\juntas'
saida_tora = 'D:\My Drive\Python-projects\Eucalipto-rachadura\python\W_Outro metodo\Face'

filenames= glob.glob(path_of_images + "/*.jpg")



for imagem in filenames:
    nome_legenda = os.path.basename(imagem)
    #img = cv2.imread('D:\My Drive\Projetos\Rachadura_Eucalipto\Analise rachadura\Fotos/1.jpg')
    img = cv2.imread(imagem)
    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ### ENCONTRAR REFERENCIA
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.inRange(gray, 200, 255)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts)
    cnts = [x for x in cnts if cv2.contourArea(x) > 1000]
    mask = np.zeros(img.shape, dtype=np.uint8)
    contorno = cv2.drawContours(mask.copy(), [max(cnts, key=cv2.contourArea)], -1, (255, 255, 255), -1)

    #cv2.imwrite(os.path.join(outfile, nome_legenda), contorno)

    ref_object = cnts[0]
    box = cv2.minAreaRect(ref_object)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="float")
    box = perspective.order_points(box)
    (tl, tr, br, bl) = box
    dist_in_pixel_larg = euclidean(tl, tr)
    dist_in_pixel_alt = euclidean(bl, tl)
    pixel_cm2 = pixel_per_cm2(dist_in_pixel_larg,dist_in_pixel_alt,3, 3)
    Um_pixel_cm2 = (1 / pixel_cm2)

    ### ENCONTRAR A ÁREA TORA

    l = np.array([0, 0, 0])
    u = np.array([255, 210, 200])
    tora = cv2.inRange(hsv, l, u)
    cnts = cv2.findContours(tora, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # Remove contours which are not large enough
    #cnts = [x for x in cnts if cv2.contourArea(x) > 1000]

    ### Remover borda 10 %
    kernel = np.ones((9, 9), np.uint8)
    mask = np.zeros(img.shape, dtype=np.uint8)
    i=1
    it = 1
    while i >=0.85:
        mask1 = cv2.drawContours(mask.copy(), [max(cnts, key=cv2.contourArea)], -1, (255, 255, 255), -1)
        mask = cv2.morphologyEx(mask1, cv2.MORPH_ERODE, kernel, iterations=it)
        valor = cv2.countNonZero(mask1[:, :, 0])
        valor1 = cv2.countNonZero(mask[:, :, 0])
        i = valor1 / valor
        it +=1
    ### Era 25,25 kernel e 7 it

    ### ENCONTRAR A ÁREA TORA
    n_rgb = cv2.medianBlur(rgb,7)
    bit_tora = cv2.bitwise_and(n_rgb,n_rgb,mask=mask[:,:,0])

    med_tora = cv2.countNonZero(mask[:,:,0]) ###area da tora
    #mask = mask[:,:,2]
    #cn = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cn = imutils.grab_contours(cn)
    #(x, y, w, h) = cv2.boundingRect(cn[0])
    #obj_tora = bit_tora[y - 50:y +50 + h, x - 50 :x+50 + w]
    #obj_tora = cv2.cvtColor(obj_tora,cv2.COLOR_RGB2BGR)

    atora = (med_tora * Um_pixel_cm2).__round__(4)
    #cv2.imwrite(os.path.join(saida_tora, nome_legenda), obj_tora)  ####Foto face

    ###ENCONTRAR RACHADURA
    bit_tora_ycc = cv2.cvtColor(bit_tora,cv2.COLOR_RGB2YCR_CB)
    v = bit_tora_ycc[:, :, 0]
    rachadura = cv2.inRange(v, 0,30)###ENCONTRAR RACHADURA 30 thresh
    bit_rachadura = cv2.bitwise_and(mask[:,:,0].copy(),rachadura)
    bit_rachadura1 = cv2.morphologyEx(bit_rachadura, cv2.MORPH_DILATE, kernel = np.ones((3, 3), np.uint8), iterations=1)
    med_rachadura = cv2.countNonZero(bit_rachadura)
    #obj_rachadura = bit_rachadura[y - 50: y + 50 + h, x - 50: x + 50 + w]
    teste = cv2.bitwise_and(rgb,rgb,mask=bit_rachadura)
    md = (med_rachadura * Um_pixel_cm2).__round__(4)
    porcentagem = ((md/atora)*100).__round__(4)
    #cv2.imwrite(os.path.join(saida, nome_legenda), obj_rachadura)  ####Foto rachadura

    ####Salvar Imagens

    #img_1 = cv2.resize(rgb,dsize=(int(img.shape[1]*0.25),int(img.shape[0]*0.25)))
    #img_1 = cv2.cvtColor(img_1,cv2.COLOR_RGB2BGR)
    #mask_1 = cv2.resize(mask1, dsize=(int(img.shape[1]*0.25),int(img.shape[0]*0.25)))
    #contorno_1 = cv2.resize(contorno, dsize=(int(img.shape[1]*0.25),int(img.shape[0]*0.25)))
    #bit_rachadura_1 = cv2.resize(bit_rachadura, dsize=(int(img.shape[1]*0.25),int(img.shape[0]*0.25)))
    #bit_rachadura_1 = cv2.cvtColor(bit_rachadura_1, cv2.COLOR_GRAY2BGR)
    #bit_rachadura_1 = cv2.putText(bit_rachadura_1,'Rachadura:{}%'.format(porcentagem),(100,100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,0,0),5)
    #im_tile = concat_tile([[img_1, mask_1], [contorno_1, bit_rachadura_1]])

    #cv2.imwrite(os.path.join(saida, nome_legenda), im_tile)  ####Fotos

    medicoes.append([nome_legenda[:-4],atora,md,porcentagem,i])

    a=(proc/(len(filenames))*100)
    print("{0:.2f} % completed".format(a))
    proc+=1

df = pd.DataFrame(medicoes)
df.columns = ['Foto','Area_Face(cm2)','Rachadura(cm2)','Porcentagem','borda_remove']
df.to_csv("medidas_15%_plus_inter(3,3).csv",index=False)
