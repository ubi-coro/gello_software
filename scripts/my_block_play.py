from dataclasses import dataclass
import numpy as np
import tyro
from dm_control import composer, viewer

from gello.agents.gello_agent import DynamixelRobotConfig
from gello.dm_control_tasks.arms.ur5e import UR5e
from gello.dm_control_tasks.manipulation.arenas.floors import Floor
from gello.dm_control_tasks.manipulation.tasks.block_play import BlockPlay

@dataclass
class Args:
    use_gello: bool = True
    port: str = "/dev/ttyDXL_gello"
    baudrate: int = 1000000

# Deine kalibrierte Konfiguration
config = DynamixelRobotConfig(
    joint_ids=(1, 2, 3, 4, 5, 6),
    joint_offsets=(-1.570796, 4.712389, 3.141593, 7.853982, 9.424778, 3.141593),
    joint_signs=(1.0, 1.0, -1.0, 1.0, 1.0, 1.0),
    gripper_config=(7, 197, 155),
    servo_types=["XC330_T288_T", "XM430_W350_T", "XM430_W350_T", "XC330_T288_T", "XC330_T288_T", "XC330_T288_T"],
)

def main(args: Args) -> None:
    # Startpose für den Roboter (nicht GELLO)
    reset_joints_left = np.deg2rad([0, -90, 90, -90, -90, 0, 0])
    
    robot = UR5e()
    # Erstelle eine Umgebung mit Boden und Blöcken
    task = BlockPlay(robot, Floor(), reset_joints=reset_joints_left[:-1])
    env = composer.Environment(task=task)

    gello = None
    if args.use_gello:
        print(f"Initializing GELLO on port {args.port} with baudrate {args.baudrate}...")
        # Erstelle den GELLO-Treiber mit deiner Config
        config.baudrate = args.baudrate 
        gello = config.make_robot(
            port=args.port, 
            start_joints=reset_joints_left
        )
        print("GELLO initialized successfully.")

    def policy(timestep) -> np.ndarray:
        if args.use_gello and gello is not None:
            joint_command = gello.get_joint_state()
            # Mapping auf Action-Space (Position Control)
            # UR5e in dm_control erwartet [joints..., gripper]
            # GELLO liefert [joints..., gripper] (0-1)
            
            # Gripper Logik: GELLO liefert 0-1, dm_control erwartet oft -1 bis 1 oder 0-255 je nach Task
            # BlockPlay Task erwartet für Gripper: 0 (offen) bis 255 (geschlossen) oder ähnlich
            # Schauen wir uns die Action Spec an:
            # action_spec = env.action_spec()
            # print(action_spec)
            
            # Einfaches Mapping:
            action = np.array(joint_command).copy()
            
            # Gripper Anpassung (letzter Wert)
            # GELLO: 0 (offen) -> 1 (zu)
            # Robotiq in MuJoCo: 0 (offen) -> 255 (zu)
            action[-1] = action[-1] * 255
            
            return action
        else:
            return np.zeros(env.action_spec().shape)

    # Starte den Viewer
    viewer.launch(env, policy=policy)

if __name__ == "__main__":
    tyro.cli(main)
