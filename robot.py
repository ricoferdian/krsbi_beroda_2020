class Robot():
    def __init__(self):
        self.is_kick_off = False
        self.is_free_kick = False
        self.is_reset = False

        self.is_dribbling_bola = False
        self.is_bola_dekat = False
        self.is_tendang_bola = False
        self.is_gawang_dekat = False

        self.my_trajectory_x = 0
        self.my_trajectory_y = 0

        self.my_absolute_x = 0
        self.my_absolute_y = 0

        self.my_gyro = 0

    def set_absolute(self, x, y):
        self.my_absolute_x = x
        self.my_absolute_xy = y

    def set_gyro(self, gyro):
        self.my_gyro = gyro

    def get_gyro(self):
        return self.my_gyro

    def set_my_trajectory(self, x, y):
        self.my_trajectory_x = x
        self.my_trajectory_y = y

    def get_my_trajectory(self):
        return self.my_trajectory_x, self.my_trajectory_y

    def kick_off(self):
        self.is_kick_off = True

    def free_kick(self):
        self.is_free_kick = True

    def reset(self):
        self.is_reset = True

    def dribble(self):
        self.is_dribbling_bola = True

    def bola_dekat(self):
        self.is_bola_dekat = True

    def tendang(self):
        self.is_tendang_bola = True

    def gawang_dekat(self):
        self.is_gawang_dekat = True

    def get_kick_off(self):
        return self.is_kick_off

    def get_reset(self):
        return self.is_reset

    def get_freekick(self):
        return self.is_free_kick

    def get_dribble(self):
        return self.is_dribbling_bola

    def get_tendang(self):
        return self.is_tendang_bola

    def get_bola_dekat(self):
        return self.is_bola_dekat

    def get_gawang_dekat(self):
        return self.is_gawang_dekat
