import serial, struct

#paramètres de la fonction vitesse_m_s, à étalonner 
struct_fmt = "!ii"
direction_prop = 1 # -1 pour les variateurs inversés ou un petit rapport correspond à une marche avant
pwm_stop_prop = 7.6
point_mort_prop = 0.21
delta_pwm_max_prop = 1.5 #pwm à laquelle on atteint la vitesse maximale

us_period = 20_000

vitesse_max_m_s_hard = 8 #vitesse que peut atteindre la voiture
vitesse_max_m_s_soft = 2 #vitesse maximale que l'on souhaite atteindre

#paramètres de départ, avec des butées très proche du centre
direction = -1 #1 pour angle_pwm_min a gauche, -1 pour angle_pwm_min à droite
#angle_pwm_min = 5.7    #min
angle_pwm_min = 4.3   #min
angle_pwm_max = 10.1
#angle_pwm_max = 9.5   #max
angle_pwm_centre= 7.5
#angle_pwm_centre= 7.3

angle_degre_max = +18 #vers la gauche
angle_degre=0

class Driver():
    
    def __init__(self):
        self.ser = serial.Serial('/dev/ttyTHS1', 115200)
        
        self.current_speed = 0
        self.current_angle = 0
    
    def set_vitesse_m_s(self, vitesse_m_s):
        if vitesse_m_s > vitesse_max_m_s_soft :
            vitesse_m_s = vitesse_max_m_s_soft
        elif vitesse_m_s < -vitesse_max_m_s_hard :
            vitesse_m_s = -vitesse_max_m_s_hard
        
        vitesse = vitesse_m_s * (delta_pwm_max_prop)/vitesse_max_m_s_hard
        if vitesse_m_s == 0 :
            new_duty_cycle = pwm_stop_prop
        elif vitesse_m_s > 0 :
            new_duty_cycle = pwm_stop_prop + direction_prop*(point_mort_prop + vitesse)
        elif vitesse_m_s < 0 :
            new_duty_cycle = (pwm_stop_prop - direction_prop*(point_mort_prop*2 - vitesse))

        self.set_propulsion(new_duty_cycle)
    
    def setCruisingSpeed(self, vitesse_km_h):
        self.set_vitesse_m_s(vitesse_km_h/3.6)
    
    def set_direction_degre(self, angle_degre):
        angle_pwm = angle_pwm_centre + direction * (angle_pwm_max - angle_pwm_min) * angle_degre /(2 * angle_degre_max )
        if angle_pwm > angle_pwm_max : 
            angle_pwm = angle_pwm_max
        if angle_pwm < angle_pwm_min :
            angle_pwm = angle_pwm_min
            
        self.set_dir(angle_pwm)
        
    def setSteeringAngle(self, angle_radians):
        self.set_direction_degre(180*angle_radians/3.14)
        
    def set_dir(self, angle):
        self.current_angle = self.duty_cycle_to_us(angle)
        self.update()
        
    def set_propulsion(self, speed):
        self.current_speed = self.duty_cycle_to_us(speed)
        self.update()
        
    def update(self):
        ret = self.ser.write(struct.pack(struct_fmt,
            self.current_angle,
            self.current_speed,
        ))
        print(f"Updated speed={self.current_speed} angle={self.current_angle} {ret=}")
        print(struct.pack(struct_fmt,self.current_angle,self.current_speed,))
        
    def duty_cycle_to_us(self, duty_cycle):
        return int((duty_cycle/100) * us_period)
    
    def stop(self):
        self.current_speed = 1500
        self.update()
        
def main2():
    import time
    drive = Driver()
    while True:
        print("direction")
        angle = int(input())
        drive.set_direction_degre(angle)
        print("vitesse")
        vitesse = float(input())
        drive.set_vitesse_m_s(vitesse)
    #drive.set_vitesse_m_s(-2)
    #time.sleep(5)
    #drive.set_vitesse_m_s(0)
    #time.sleep(2)
    #drive.set_vitesse_m_s(2)
    #time.sleep(2)
    drive.set_direction_degre(90)
    time.sleep(0.1)
    drive.set_direction_degre(0)
    time.sleep(0.1)
    drive.set_direction_degre(-90)
    time.sleep(0.1)
    drive.set_direction_degre(0)
    time.sleep(0.1)
    drive.set_direction_degre(90)
    time.sleep(0.1)
    drive.set_direction_degre(0)


if __name__ == "__main__":
    main2()
