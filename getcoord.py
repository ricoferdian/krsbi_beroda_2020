from helper_new import *

cameraCenterX = 343
cameraCenterY = 238
def getcoordinate(arr_objects):
    for index, object in enumerate(arr_objects):
        print(object)
        arr_objects[index]['centerX'] = (object['x2'] + object['x1'])/2
        arr_objects[index]['centerY'] = (object['y2'] + object['y1'])/2
    return getDistance(arr_objects)

def getDistance(arr_objects_coord):
    for index, object in enumerate(arr_objects_coord):
        arr_objects_coord[index]['realPxObj'] = distancePixel(cameraCenterX, cameraCenterY, object['centerX'], object['centerY'])
        arr_objects_coord[index]['tetaObj'] = teta(cameraCenterX, cameraCenterY, object['centerX'], object['centerY'])

        arr_objects_coord[index]['realDistance'] = distPix2real(arr_objects_coord[index]['realPxObj'])
        arr_objects_coord[index]['realDistanceY'] = -arr_objects_coord[index]['realDistance'] * cos(radians(arr_objects_coord[index]['tetaObj']))
        arr_objects_coord[index]['realDistanceX'] = arr_objects_coord[index]['realDistance'] * sin(radians(arr_objects_coord[index]['tetaObj']))
    return arr_objects_coord




