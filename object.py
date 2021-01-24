from helper_new import *


class Object():
    def __init__(self, camera_c_x, camera_c_y):
        # Set center kamera
        self.camera_c_x = camera_c_x
        self.camera_c_y = camera_c_y

        # Posisi object dalam image
        self.center_x = 0
        self.center_y = 0

        # Posisi object yang telah dikonversi dalam jarak real
        self.real_distance_x = 0
        self.real_distance_y = 0
        self.theta = 0

    def set_center_objects(self, x1, y1, x2, y2):
        self.center_x = x2 + x1 / 2
        self.center_y = y2 + y1 / 2

        # Langsung get real distance
        self.get_real_distance()

    def get_real_distance(self):
        real_distance_px = distancePixel(self.camera_c_x, self.camera_c_y, self.center_x, self.center_y)
        self.theta = -1*teta(self.camera_c_x, self.camera_c_y, self.center_x, self.center_y)

        real_distance = distPix2real(real_distance_px)
        self.real_distance_x = real_distance * cos(radians(self.theta))
        self.real_distance_y = real_distance * sin(radians(self.theta))

    def get_x_y_theta(self):
        return self.real_distance_x, self.real_distance_y, self.theta
