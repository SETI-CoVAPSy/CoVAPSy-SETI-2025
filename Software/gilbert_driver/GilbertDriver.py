import serial
import struct
from typing import Self, Literal
from types import TracebackType


class GilbertDriver:
    """Driver for the Gilbert car."""
    # === Calibration parameters ===
    STEERING_DIRECTION: Literal[1, -1] = 1 # 1 for clockwise (min duty = left), -1 for counter-clockwise (min duty = right)
    PWM_PERIOD_US: int = 20_000            # PWM period (in microseconds)

    PWM_DUTY_ANGLE_MAX: float = 10.5       # PWM duty cycle (%) for maximum steering angle (deg)
    PWM_DUTY_ANGLE_MIN: float = 4.5        # PWM duty cycle (%) for minimum steering angle (deg)
    # PWM_DUTY_ANGLE_NEUTRAL: float = 7.5  # PWM duty cycle (%) for neutral (0 deg) steering
    PWM_DUTY_SPEED_NEUTRAL: float = 7.6    # PWM duty cycle (%) for neutral (0) speed
    PWM_DUTY_SPEED_MAX: float = 9.1        # PWM duty cycle (%) for maximum forward speed

    SPEED_MAX: float = 8.0                 # Maximum speed (at PWM_DUTY_SPEED_MAX) (m/s)
    SPEED_LIMIT_FORWARD: float = 2.0       # Speed limit for our application (m/s)
    SPEED_LIMIT_REVERSE: float = -8.0      # Reverse speed limit (m/s)
    ANGLE_LIMIT_DEG: float = 18.0          # Maximum steering angle (degrees)

    # === Misc ===
    CMD_PREFIX = "!ii"  # First object in sent struct message

    def __init__(
        self,
        serial_port: str = "/dev/ttyTHS1",
        serial_baud: int = 115200,
        verbose: bool = False,
        auto_open: bool = True,
    ) -> None:
        """Driver for the Gilbert car.

        Args:
            serial_port (str): Serial port for the STM32 serial connection. Defaults to '/dev/ttyTHS1'.
            serial_baud (int): Baud rate for the serial connection. Defaults to 115200.
            auto_open (bool): Whether to automatically open the serial connection on initialization. Defaults to True.
            verbose (bool): Whether to print debug information. Defaults to False.

        Note: auto_open is True by default for convenience, but using a context manager (with ... as ..:) is recommended to ensure correct opening and closing of the serial connection.
        """
        self._verbose = verbose
        if self._verbose:
            print("Initializing GilbertDriver (verbose on)...")

        # Configure serial connection without opening it yet
        self.serial = serial.Serial()
        self.serial.port = serial_port
        self.serial.baudrate = serial_baud

        self.speed_mps = 0.0  # Current speed (m/s)
        self.angle_deg = 0.0  # Current steering angle (degrees)

        if auto_open:
            self.open()

    def open(self) -> None:
        """Open the serial connection to the STM32 controller."""
        self.serial.port = "/dev/ttyTHS1"
        self.serial.baudrate = 115200
        self.serial.open()
        if self._verbose:
            print(f"Serial connection opened at {self.serial.port} with {self.serial.baudrate}.")

    def close(self) -> None:
        """Close the serial connection to the STM32 controller."""
        self.serial.close()
        if self._verbose:
            print("Serial connection closed.")

    def __enter__(self) -> Self:
        """Opens the serial connection when entering context."""
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        traceback: TracebackType,
    ) -> None:
        """Closes cleanly the serial connection when exiting context."""
        self.close()
    
    @staticmethod
    def duty_cycle_to_us(duty_cycle):
        return int((duty_cycle/100) * GilbertDriver.PWM_PERIOD_US)

    def send_command(self) -> None:
        """Send set speed and angle to the STM32 controller. Will use internal state."""
        # Retrieve corresponding duty cycles
        pwm_duty_angle_centre = self.PWM_DUTY_ANGLE_MAX + self.PWM_DUTY_ANGLE_MIN / 2
        pwm_duty_angle = pwm_duty_angle_centre + self.STEERING_DIRECTION * (self.PWM_DUTY_ANGLE_MAX - self.PWM_DUTY_ANGLE_MIN) * self.angle_deg / (2 * self.ANGLE_LIMIT_DEG)
        if pwm_duty_angle > self.PWM_DUTY_ANGLE_MAX:
            pwm_duty_angle = self.PWM_DUTY_ANGLE_MAX
        elif pwm_duty_angle < self.PWM_DUTY_ANGLE_MIN:
            pwm_duty_angle = self.PWM_DUTY_ANGLE_MIN
        
        # Speed already limited in set_speed_mps
        pwm_duty_speed_centre = self.PWM_DUTY_SPEED_NEUTRAL
        pwm_duty_speed = pwm_duty_speed_centre + (self.PWM_DUTY_SPEED_MAX - pwm_duty_speed_centre) * self.speed_mps / self.SPEED_MAX

        # Get microsecond values
        angle_us = GilbertDriver.duty_cycle_to_us(pwm_duty_angle)
        speed_us = GilbertDriver.duty_cycle_to_us(pwm_duty_speed)

        msg = struct.pack(
            GilbertDriver.CMD_PREFIX,
            angle_us,
            speed_us,
        )
        ret = self.serial.write(msg)
        if self._verbose:
            print(f"Updated speed={self.speed_mps} angle={self.angle_deg} ({ret=})")

    def set_speed_mps(self, speed_mps: float) -> None:
        """Set the speed in meters per second (algebraic).

        Args:
            speed_mps (float): Desired speed in meters per second.
        """
        self.speed_mps = speed_mps
        if self.speed_mps > self.SPEED_LIMIT_FORWARD:
            self.speed_mps = self.SPEED_LIMIT_FORWARD
        elif self.speed_mps < self.SPEED_LIMIT_REVERSE:
            self.speed_mps = self.SPEED_LIMIT_REVERSE
        self.send_command()
    
    def set_speed_kmph(self, speed_kmh: float) -> None:
        """Set the speed in kilometers per hour (algebraic).

        Args:
            speed_kmh (float): Desired speed in kilometers per hour.
        """
        self.set_speed_mps(speed_kmh / 3.6)

    def brake(self) -> None:
        """Set speed to zero (brake)."""
        self.set_speed_mps(0.0)

    def set_steering_angle_deg(self, angle_deg: float) -> None:
        """Set the steering angle in degrees.

        Args:
            angle_deg (float): Desired steering angle in degrees.
        """
        self.angle_deg = angle_deg
        self.send_command()

    def neutral(self) -> None:
        """Set speed and steering angle to neutral (0)."""
        self.speed_mps = 0.0
        self.angle_deg = 0.0
        self.send_command()

if __name__ == '__main__':
    with GilbertDriver(verbose=True) as gilbert:
        while True:
            for angle in [0, -20, 0, 20]*2:
                print(f"Setting angle to {angle} degrees")
                gilbert.set_steering_angle_deg(angle)
            
            for speed in [0, 1.0, 0, -1.0]*2:
                print(f"Setting speed to {speed} m/s")
                gilbert.set_speed_mps(speed)
            
            gilbert.neutral()
