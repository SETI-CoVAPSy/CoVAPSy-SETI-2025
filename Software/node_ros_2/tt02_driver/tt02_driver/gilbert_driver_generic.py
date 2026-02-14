"""
Generic driver for the Gilbert car (speed and steering angle control), defined the base class.
"""

from typing_extensions import Self
from types import TracebackType
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from numpy import uint8

class GilbertDriverGeneric(ABC):
    """Generic driver for the Gilbert car."""

    PWM_PERIOD_US: int = 20_000  # PWM period (in microseconds)
    SPEED_LIMIT_FORWARD: float = 2.0  # Speed limit for our application (m/s)
    SPEED_LIMIT_REVERSE: float = -8.0  # Reverse speed limit (m/s)
    ANGLE_LIMIT_DEG: float = 4.0  # Maximum steering angle (degrees)

    CAMERA_FOV_DEG: float = 120.0 # FoV of the camera (degrees)

    def duty_cycle_to_us(self, duty_cycle):
        """Convert duty cycle [0.0-100.0] to corresponding duration in µs."""
        return int((duty_cycle / 100) * self.PWM_PERIOD_US)

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

    def neutral(self) -> None:
        """Set speed and steering angle to neutral (0)."""
        self.speed_mps = 0.0
        self.angle_deg = 0.0
        self.send_command()

    def set_steering_angle_deg(self, angle_deg: float) -> None:
        """Set the steering angle in degrees.

        Args:
            angle_deg (float): Desired steering angle in degrees.
        """
        if angle_deg < -self.ANGLE_LIMIT_DEG:
            angle_deg = -self.ANGLE_LIMIT_DEG
        elif angle_deg > self.ANGLE_LIMIT_DEG:
            angle_deg = self.ANGLE_LIMIT_DEG
        self.angle_deg = angle_deg
        self.send_command()

    def __enter__(self) -> Self:
        """Opens the relevant connections when entering context."""
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        traceback: TracebackType,
    ) -> None:
        """Closes cleanly the relevant connections when exiting context."""
        self.close()

    def open(self) -> None:
        """Open relevant connections."""
        pass

    def close(self) -> None:
        """Close relevant connections."""
        pass

    def log(self, message: str) -> None:
        """Log a message (for debugging)."""
        print(f"[{self.__class__.__name__}] {message}")

    @abstractmethod
    def send_command(self) -> None:
        """Send set speed and angle of the car. Will use internal state."""
        raise NotImplementedError()

    @abstractmethod
    def get_camera_frame(self) -> None | NDArray[uint8]:
        """Capture a frame from the camera (if enabled)."""
        raise NotImplementedError()