import time
import numpy as np
from gello.zmq_core.robot_node import ZMQClientRobot

def main():
    # Connect to the simulation (default port 6001)
    robot = ZMQClientRobot(host="127.0.0.1", port=6001)
    
    print("Connecting to robot...")
    try:
        # Read a few times to clear buffer
        for _ in range(5):
            joints = robot.get_joint_state()
            time.sleep(0.1)
        
        print("\n--- Current Robot Joint Angles ---")
        print(f"Radians: {np.array2string(joints, precision=4, separator=', ')}")
        print(f"Degrees: {np.array2string(np.degrees(joints), precision=1, separator=', ')}")
        print("----------------------------------")
        print("\nCopy the 'Radians' list into your config file under 'initial_match_joint_pos'.")
        
    except Exception as e:
        print(f"Error connecting to robot: {e}")
        print("Make sure the simulation is running!")

if __name__ == "__main__":
    main()
