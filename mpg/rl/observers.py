import tf_agents as tfa


def printer_observer(trajectory: tfa.trajectories.Trajectory):
    print(trajectory)

def printer_state_observer(trajectory: tfa.trajectories.Trajectory):
    print(trajectory.observation["state"])


class PrinterObserver:
    def __init__(self, state:bool=True, action:bool=True, name="PrinterObserver"):
        self.name = name
        self.state=state
        self.action=action
        if not state and not action:
            raise ValueError("At least one of state and action must be true")

    def __call__(self, trajectory: tfa.trajectories.Trajectory):
        if self.state and self.action:
            print(f"S:{trajectory.observation['state']}, A:{trajectory.action}")
        elif self.state:
            print(f"S:{trajectory.observation['state']}")
        elif self.action:
            print(f"A:{trajectory.action}")