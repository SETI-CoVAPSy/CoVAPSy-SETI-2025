"""
Driver for the Gilbert car (speed and steering angle control) compatible with the Webots simulator.

Copyright 1996-2022 Cyberbotics Ltd.

Controle de la voiture TT-02 simulateur CoVAPSy pour Webots 2023b
Inspiré de vehicle_driver_altino controller
Kévin Hoarau, Anthony Juton, Bastien Lhopitallier, Martin Taynaud
juillet 2023

Driver basé sur l'implémentation décrite ci-dessus pour le projet CoVAPSy.
"""

from vehicle import Driver as WebotsDriver

from typing import Literal
from typing_extensions import override
from .gilbert_driver_generic import GilbertDriverGeneric
import numpy as np
from numpy.typing import NDArray
from numpy import uint8

class GilbertDriverWebots(GilbertDriverGeneric):
    """Driver for the Gilbert car."""

    # === Calibration parameters ===
    STEERING_DIRECTION: Literal[1, -1] = (
        1  # 1 for clockwise (min duty = left), -1 for counter-clockwise (min duty = right)
    )
    SPEED_MAX: float = 8.0  # Maximum speed (at PWM_DUTY_SPEED_MAX) (m/s)
    SPEED_LIMIT_FORWARD: float = 2.0  # Speed limit for our application (m/s)
    SPEED_LIMIT_REVERSE: float = -8.0  # Reverse speed limit (m/s)
    ANGLE_LIMIT_DEG: float = 18.0  # Maximum steering angle (degrees)

    CAMERA_FOV_DEG: float = 120.0 # FoV of the camera (degrees)

    def __init__(
        self,
        use_camera: bool = False,
        auto_open: bool = True,
        verbose: bool = False,
        camera_name: str = "camera_gilbert",
    ) -> None:
        """Driver for the Gilbert car (software).

        Args:
            use_camera (bool): Whether to use the camera. Defaults to False.
            auto_open (bool): Whether to automatically open the connection and start camera on initialization. Defaults to True.
            verbose (bool): Whether to print debug information. Defaults to False.
            camera_name (str): Name of the camera in Webots. Defaults to "camera".
        
        Note: auto_open is True by default for convenience, but using a context manager (with ... as ..:) is recommended to ensure correct opening and closing of the serial connection.
        """
        self._verbose = verbose
        if self._verbose:
            self.log("Initializing GilbertDriver (simulation) (verbose on)...")

        self._driver: WebotsDriver | None = None
        self.speed_mps = 0.0  # Current speed (m/s)
        self.angle_deg = 0.0  # Current steering angle (degrees)

        self._use_camera = use_camera
        self._camera_name = camera_name
        self.camera = None

        if auto_open:
            self.open()

    @override
    def open(self) -> None:
        """Open relevant connections."""
        self._driver = WebotsDriver()
        if self._use_camera:
            from controller import Camera
            self.camera = Camera(self._camera_name)
            self.camera.enable(int(round(self._driver.getBasicTimeStep())))

    @override
    def close(self) -> None:
        """Close the relevant connection."""
        if self.camera:
            self.camera.disable()
        del self._driver
        self._driver = None

    @override
    def send_command(self) -> None:
        """Send set speed and angle to the Webots controller. Will use internal state."""
        if self._driver is None:
            raise RuntimeError("Driver not opened. Call open() before sending commands.")
        self._driver.setSteeringAngle(self.angle_deg * 3.14/180 * self.STEERING_DIRECTION)
        self._driver.setCruisingSpeed(self.speed_mps * 3.6)

        if self._verbose:
            self.log(f"Updated speed={self.speed_mps}m/s angle={self.angle_deg}deg")

    def get_driver(self) -> WebotsDriver:
        """Retrieve Webots driver object."""
        if self._driver is None:
            raise RuntimeError("Driver not opened. Call open() before getting the driver.")
        return self._driver

    @override
    def get_camera_frame(self) -> None | NDArray[uint8]:
        """Capture a frame from the camera (if enabled)."""
        if not self.camera:
            return None
        
        return np.array(self.camera.getImageArray(), dtype=np.uint8)