from typing import Any


class TrajectoryView:
    def __init__(self, trajectory: list[Any]):
        self.trajectory = trajectory

    @property
    def states(self) -> list[Any]:
        return self.trajectory[::2]  # Even indices are state infos

    @property
    def actions(self) -> list[Any]:
        return self.trajectory[1::2]  # Odd indices are actions
