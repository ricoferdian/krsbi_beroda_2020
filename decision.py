def decision_making(object, isDribblingBola, isEndpointInit, isNemuBola, bolaLastSeenX, bolaLastSeenY):
    if (object['label'] == 'bola'):
        isNemuBola = True
        if (not isDribblingBola):
            print('AKU NYARI BOLA', object)
            isNyariBola = True
            isTendangBola = False
            end = {}
            end['x'] = object['x']
            end['y'] = object['y']
            bolaLastSeenX = object['x']
            bolaLastSeenY = object['y']
            tetaBall = object['tetaObj']
            realDistanceX = object['real_distance_x']
            realDistanceY = object['realDistanceY']
            if (not isEndpointInit):
                isEndpointInit = True
            if (realDistanceY < 250):
                print('BOLA SUDAH DEKAT')
                isBolaDekat = True
        else:
            if (not isEndpointInit):
                end = None
                isEndpointInit = True
    elif (object['label'] == 'gawang' and isDribblingBola):
        # Cari gawangs
        print('AKU NYARI GAWANG', object)
        end = {}
        end['x'] = object['x']
        end['y'] = object['y']
        isEndpointInit = True
        tetaBall = object['tetaObj']
        realDistanceX = object['real_distance_x'] * 0.8
        realDistanceY = object['realDistanceY'] * 0.8
        # JIKA GAWANG DEKAT, TENDANG
        if (isDribblingBola and realDistanceY < 400):
            isTendangBola = True
        # elif(realDistanceY<400):
        #     realDistanceY = 0
        #     real_distance_x = 0

        else:
            if (not isEndpointInit):
                end = None
                isEndpointInit = True
    # elif (object['label'] == 'robot'):
    #     if (isDribblingBola or (strategyState==arrayStrategy[2] or strategyState==arrayStrategy[5] or strategyState==arrayStrategy[9])):
    #         # Cari gawang
    #         print('AKU NYARI TEMENKU DIMANA', object)
    #         end = {}
    #         end['x'] = object['x']
    #         end['y'] = object['y']
    #         isEndpointInit = True
    #         tetaBall = object['tetaObj']
    #         real_distance_x = object['real_distance_x']*0.9
    #         realDistanceY = object['realDistanceY']*0.9
    #         #JIKA ROBOT DEKAT, TENDANG
    #         if(not isDribblingBola):
    #             realDistanceY = 0
    #             real_distance_x = 0
    #         if(isDribblingBola and realDistanceY<300):
    #             realDistanceY = 0
    #             isTendangBola = True
    #     else:
    #         if (not isEndpointInit):
    #             end = None
    #             isEndpointInit = True
    # elif(object['label'] == 'obstacle'):
    #     print('ADA OBSTACLE', object)
    #     obstacle = {}
    #     obstacle['x'] = object['x']*0.9
    #     obstacle['y'] = object['y']*0.9
    #     obstacleGridLoc = getGridLocationFromCoord(obstacle,splitSizeGrid)
    #     if(obstacleGridLoc[0]>11):
    #         obstacleGridLoc[0] = 11
    #     elif(obstacleGridLoc[0]<0):
    #         obstacleGridLoc[0] = 0
    #     if(obstacleGridLoc[1]>8):
    #         obstacleGridLoc[1] = 8
    #     elif(obstacleGridLoc[1]<0):
    #         obstacleGridLoc[1] = 0
    #
    #     # print("obstacleGridLoc",obstacleGridLoc)
    #     matrix[obstacleGridLoc[1]][obstacleGridLoc[0]] = 0
    #     if(not isNemuBola or not isDribblingBola):
    #         if(not isEndpointInit):
    #             end = None
    if (not isNemuBola):
        # Berputar-putar sampai melihat bola
        print('AKU BINGUNG BOLANYA DIMANA')
        end = {}
        end['x'] = bolaLastSeenX
        end['y'] = bolaLastSeenY
        realDistanceX = bolaLastSeenX
        realDistanceY = bolaLastSeenY
    else:
        if (not isEndpointInit):
            end = None

    return realDistanceX, realDistanceY, end,