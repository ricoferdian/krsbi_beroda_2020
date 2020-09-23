from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from math import *

#Generate grid lapangan (center grid saja yang disimpan untuk pathfinding)
def gridGenerator(maxX,maxY,splitSize):
    arrayGrid = []
    x = 0
    while x < maxX:
        y = 0
        rowArrayGrid = []
        while y < maxY:
            colArrayGrid = [x+(splitSize/2),y+(splitSize/2)]
            rowArrayGrid.append(colArrayGrid)
            y += splitSize
        arrayGrid.append(rowArrayGrid)
        x += splitSize
    return arrayGrid

def getGridLocationFromCoord(coord,splitSize):
    x =  coord['x']
    y = coord['y']
    #Get alamat grid mana dia
    modCoordX = floor(x/splitSize)
    modCoordY = floor(y/splitSize)
    return [modCoordX,modCoordY]

def rotateMatrix(x,y,angle):
    x1 = (cos(radians(angle))*x)-(sin(radians(angle))*y)
    y1 = (sin(radians(angle))*x)+(cos(radians(angle))*y)
    return x1,y1

def findPathRobot(startGridLoc, matrixGridLapangan, endGridLoc):
    grid = Grid(matrix=matrixGridLapangan)

    # Start always updating tergantung value yang diberikan dari master
    # Data start diambil dari serial
    start = grid.node(startGridLoc[0], startGridLoc[1])
    # Data end diambil dari kamera
    # End always update hanya untuk bola, kalau gawang tetap tapi tetap perlu update
    end = grid.node(endGridLoc[0], endGridLoc[1])
    # Hasil path diberikan ke master lagi

    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    paths, runs = finder.find_path(start, end, grid)

    return paths