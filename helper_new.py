from math import *

setRotateOnCam = -90

# fungsi jarak (Jaraknya miring kamera)
def distPix2real(miring):
    if(miring > 25):
        return sqrt(abs(pow((miring * 1.6560563) + 29.943605, 2) - 6400))
    else:
        return 80

# def distPix2real(rPix):
#     return (0.07423670095971158*rPix*rPix)-(17.404005277931052*rPix)+1116.5834444460588

# def distPix2real(rPix):
#     return (rPix * 2.1059295 - 133.71944)

# def distPix2real(rPix):
#     rReal = (3 * (10 ** -34) * (rPix ** 6)) - (3 * (10 ** -10) * (rPix ** 5)) + (1 * (10 ** -7) * (rPix ** 4)) - \
#             (2 * (10 ** -5) * (rPix ** 3)) - (0.0025 * (rPix ** 2)) + (1.1661 * rPix) + 43.035
#     return rReal

# fungsi jarak
def distancePixel(tengah_x, tengah_y, titik_x, titik_y):
    delx = titik_x - float(tengah_x)
    dely = titik_y - float(tengah_y)
    j_rad = sqrt(pow(delx, 2) + pow(dely, 2))
    return j_rad

# fungsi teta
def teta(tengah_x, tengah_y, titik_x, titik_y):
    delx = -(float(tengah_x) - titik_x)
    dely = float(tengah_y) - titik_y
    oteta = atan2(delx, dely)
    oteta = oteta / pi * 180
    oteta = oteta + setRotateOnCam
    if oteta > 180:
        oteta = oteta - 360
    elif oteta < -180:
        oteta = oteta + 360
    return oteta
