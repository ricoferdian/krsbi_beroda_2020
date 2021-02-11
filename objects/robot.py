class Robot():
    def __init__(self):
        # Variabel destination object yang dituju robot
        self.dest_x = 0
        self.dest_y = 0
        self.dest_theta = 0

        # Variabel aksi utama
        self.is_kick_off = False
        self.is_free_kick = False
        self.is_reset = False

        # Variabel status apa yang mau dilakukan
        self.is_dribbling_bola = False
        self.is_bola_dekat = False
        self.is_tendang_bola = False
        self.is_gawang_dekat = False
        self.is_dest_found = False

        # Variabel trajectory dari mikro
        self.my_trajectory_x = 0
        self.my_trajectory_y = 0

        # Variabel gyro dari mikro
        self.my_gyro = 0

        # Variabel absolut lapangan (belum dihitung)
        self.my_absolute_x = 0
        self.my_absolute_y = 0

    def set_dest(self, x, y, theta):
        self.dest_x = x
        self.dest_y = y
        self.dest_theta = theta
        self.is_dest_found = True

    def get_dest(self):
        return self.dest_x, self.dest_y, self.dest_theta

    # Fungsi untuk dipanggil saat parsing data ke mikro
    def get_destination(self):
        return self.dest_x, self.dest_y, self.dest_theta, self.is_tendang_bola, self.is_bola_dekat, self.is_reset

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

    def set_kick_off(self, kickoff):
        self.is_kick_off = kickoff

    def set_free_kick(self, freekick):
        self.is_free_kick = freekick

    def set_reset(self, reset):
        self.is_reset = reset

    def set_dribble(self, dribble):
        self.is_dribbling_bola = dribble

    def set_bola_dekat(self, bola_dekat):
        self.is_bola_dekat = bola_dekat

    def set_tendang(self, tendang):
        self.is_tendang_bola = tendang

    def set_gawang_dekat(self, gawangdekat):
        self.is_gawang_dekat = gawangdekat

    def get_kickoff(self):
        return self.is_kick_off

    def get_dest_status(self):
        return self.is_dest_found

    def set_dest_status(self, status):
        self.is_dest_found = status

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
