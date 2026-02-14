"""
Driver for the Gilbert car (speed and steering angle control) compatible with the physical car.
"""

import serial
import struct
from typing import Literal, Optional, Dict, Any
from typing_extensions import override
from .gilbert_driver_generic import GilbertDriverGeneric
from numpy.typing import NDArray
from numpy import uint8

class GilbertDriverHardware(GilbertDriverGeneric):
    """Driver for the Gilbert car (hardware)."""

    # === Calibration parameters ===
    STEERING_DIRECTION: Literal[1, -1] = (
        1  # 1 for clockwise (min duty = left), -1 for counter-clockwise (min duty = right)
    )
    PWM_PERIOD_US: int = 20_000  # PWM period (in microseconds)

    PWM_DUTY_ANGLE_MAX: float = (
        10.5  # PWM duty cycle (%) for maximum steering angle (deg)
    )
    PWM_DUTY_ANGLE_MIN: float = (
        4.5  # PWM duty cycle (%) for minimum steering angle (deg)
    )
    # PWM_DUTY_ANGLE_NEUTRAL: float = 7.5  # PWM duty cycle (%) for neutral (0 deg) steering
    PWM_DUTY_SPEED_NEUTRAL: float = 7.6  # PWM duty cycle (%) for neutral (0) speed
    PWM_DUTY_SPEED_MAX: float = 9.1  # PWM duty cycle (%) for maximum forward speed

    SPEED_MAX: float = 8.0  # Maximum speed (at PWM_DUTY_SPEED_MAX) (m/s)
    SPEED_LIMIT_FORWARD: float = 2.0  # Speed limit for our application (m/s)
    SPEED_LIMIT_REVERSE: float = -8.0  # Reverse speed limit (m/s)
    ANGLE_LIMIT_DEG: float = 18.0  # Maximum steering angle (degrees)

    CAMERA_FOV_DEG: float = 120.0 # FoV of the camera (degrees)
    
    # === Misc ===
    CMD_PREFIX = "!ii"  # First object in sent struct message

    def __init__(
        self,
        serial_port: str = "/dev/ttyTHS1",
        serial_baud: int = 115200,
        auto_open: bool = True,
        use_camera: bool = False,
        verbose: bool = False,
    ) -> None:
        """Driver for the Gilbert car.

        Args:
            serial_port (str): Serial port for the STM32 serial connection. Defaults to '/dev/ttyTHS1'.
            serial_baud (int): Baud rate for the serial connection. Defaults to 115200.
            use_camera (bool): Whether to use the camera. Defaults to False.
            auto_open (bool): Whether to automatically open the serial connection and start camera on initialization. Defaults to True.
            verbose (bool): Whether to print debug information. Defaults to False.

        Note: auto_open is True by default for convenience, but using a context manager (with ... as ..:) is recommended to ensure correct opening and closing of the serial connection.
        """
        self._verbose = verbose
        if self._verbose:
            self.log("Initializing GilbertDriver (verbose on)...")

        # Configure serial connection without opening it yet
        self.serial = serial.Serial()
        self.serial.port = serial_port
        self.serial.baudrate = serial_baud

        # Camera
        self.use_camera = use_camera
        self.camera = None
        self._robot = None # Used for init
        self.camera_config = None
        if use_camera:
            from picamera2 import Picamera2
            from controller import Robot
            self.camera = Picamera2()
            self._robot = Robot()

            # Main config
            self.camera_config = self.camera.create_still_configuration(
                # main={"size": (640, 480)} # Customize if needed
            )
            self.camera_config["format"] = "RGB888"  # Ensure RGB       

            self.camera.configure(self.camera_config)

            # Misc config
            # self.camera.set_controls({"ExposureTime": 10000, "AnalogueGain": 1.0})
            # self.camera.set_controls({"AfMode": controls.AfModeEnum.Continuous}) # Continuous autofocus

        if auto_open:
            self.open()

        self.speed_mps = 0.0  # Current speed (m/s)
        self.angle_deg = 0.0  # Current steering angle (degrees)

    @override
    def open(self) -> None:
        """Open the serial connection to the STM32 controller."""
        self.serial.port = "/dev/ttyTHS1"
        self.serial.baudrate = 115200
        self.serial.open()
        if self._verbose:
            self.log(
                f"Serial connection opened at {self.serial.port} with {self.serial.baudrate}."
            )
        # Also ensure camera is started when opening hardware
        if self.camera:
            try:
                self.camera.start()
                if self._verbose:
                    self.log("Picamera2 started on open()")
            except Exception as exc:
                if self._verbose:
                    self.log(f"Failed to start Picamera2 on open(): {exc}")

    @override
    def close(self) -> None:
        """Close the serial connection to the STM32 controller."""
        # Stop camera first if it was started by this driver
        self.serial.close()
        if self._verbose:
            self.log("Serial connection closed.")

        if self.camera:
            try:
                self.camera.stop()
                if self._verbose:
                    self.log("Picamera2 stopped on close()")
            except Exception as exc:
                if self._verbose:
                    self.log(f"Failed to stop Picamera2 on close(): {exc}")

    @override
    def send_command(self) -> None:
        """Send set speed and angle to the STM32 controller. Will use internal state."""
        # Retrieve corresponding duty cycles
        pwm_duty_angle_centre = self.PWM_DUTY_ANGLE_MAX + self.PWM_DUTY_ANGLE_MIN / 2
        pwm_duty_angle = pwm_duty_angle_centre + self.STEERING_DIRECTION * (
            self.PWM_DUTY_ANGLE_MAX - self.PWM_DUTY_ANGLE_MIN
        ) * self.angle_deg / (2 * self.ANGLE_LIMIT_DEG)
        if pwm_duty_angle > self.PWM_DUTY_ANGLE_MAX:
            pwm_duty_angle = self.PWM_DUTY_ANGLE_MAX
        elif pwm_duty_angle < self.PWM_DUTY_ANGLE_MIN:
            pwm_duty_angle = self.PWM_DUTY_ANGLE_MIN

        # Speed already limited in set_speed_mps
        pwm_duty_speed_centre = self.PWM_DUTY_SPEED_NEUTRAL
        pwm_duty_speed = (
            pwm_duty_speed_centre
            + (self.PWM_DUTY_SPEED_MAX - pwm_duty_speed_centre)
            * self.speed_mps
            / self.SPEED_MAX
        )

        # Get microsecond values
        angle_us = self.duty_cycle_to_us(pwm_duty_angle)
        speed_us = self.duty_cycle_to_us(pwm_duty_speed)

        msg = struct.pack(
            self.CMD_PREFIX,
            angle_us,
            speed_us,
        )
        ret = self.serial.write(msg)
        if self._verbose:
            self.log(f"Updated speed={self.speed_mps}m/s angle={self.angle_deg}deg ({ret=})")

    @override
    def get_camera_frame(self) -> None | NDArray[uint8]:
        """Capture a frame from the camera (if enabled)."""
        if self.camera is None:
            return None
        
        return self.camera.capture_array()

if __name__ == "__main__":
    with GilbertDriverHardware(verbose=True, auto_open=False) as gilbert:
        while True:
            for angle in [0, -20, 0, 20] * 2:
                gilbert.log(f"Setting angle to {angle} degrees")
                gilbert.set_steering_angle_deg(angle)

            for speed in [0, 1.0, 0, -1.0] * 2:
                gilbert.log(f"Setting speed to {speed} m/s")
                gilbert.set_speed_mps(speed)

            gilbert.neutral()
