"""
Implémentation naive (traduction directe) de controller_jaune.py
"""

from gilbert_driver.gilbert_driver_webots import GilbertDriverWebots
from controller import Lidar

if __name__ == '__main__':
    gilbert = GilbertDriverWebots(verbose=True)
    gilbert.STEERING_DIRECTION = -1 # Inverted
    webots_driver = gilbert.get_driver()
    basicTimeStep = int(webots_driver.getBasicTimeStep())
    sensorTimeStep = 4 * basicTimeStep

    #Lidar
    lidar = Lidar("RpLidarA2")
    lidar.enable(sensorTimeStep)
    lidar.enablePointCloud() 

    #clavier
    keyboard = webots_driver.getKeyboard()
    keyboard.enable(sensorTimeStep)

    tableau_lidar_mm=[0]*360

    modeAuto = False
    print("cliquer sur la vue 3D pour commencer")
    print("a pour mode auto (pas de mode manuel sur TT02_jaune), n pour stop")

    while webots_driver.step() != -1:
        # Gestion touches clavier
        while True:
            currentKey = keyboard.getKey()

            if currentKey == -1:
                break
        
            elif currentKey == ord('n') or currentKey == ord('N'):
                if modeAuto :
                    modeAuto = False
                    print("--------Modes Auto TT-02 jaune Désactivé-------")
            elif currentKey == ord('a') or currentKey == ord('A'):
                if not modeAuto : 
                    modeAuto = True
                    print("------------Mode Auto TT-02 jaune Activé-----------------")


        #acquisition des donnees du lidar
        donnees_lidar_brutes = lidar.getRangeImage()
        for i in range(360) :
            if (donnees_lidar_brutes[-i]>0) and (donnees_lidar_brutes[-i]<20) :
                tableau_lidar_mm[i-180] = 1000*donnees_lidar_brutes[-i]
            else :
                tableau_lidar_mm[i-180] = 0

        if not modeAuto:
            gilbert.set_speed_mps(0)
            gilbert.set_steering_angle_deg(0)
            
        if modeAuto:
        ########################################################
        # Programme etudiant avec
        #    - le tableau tableau_lidar_mm
        #    - la fonction set_direction_degre(...)
        #    - la fonction set_vitesse_m_s(...)
        #    - la fonction recule()
        #######################################################

            #un secteur par tranche de 20° donc 10 secteurs numérotés de 0 à 9    
            angle_degre = 0.02*(tableau_lidar_mm[60]-tableau_lidar_mm[-60])
            gilbert.set_steering_angle_deg(angle_degre)
            vitesse_m_s = 0.5
            gilbert.set_speed_mps(vitesse_m_s)

        #########################################################


