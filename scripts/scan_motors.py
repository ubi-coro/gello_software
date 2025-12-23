from dynamixel_sdk import *
import os

# Control table address
ADDR_TORQUE_ENABLE      = 64               # Control table address is different in Dynamixel model
ADDR_GOAL_POSITION      = 116
ADDR_PRESENT_POSITION   = 132

# Protocol version
PROTOCOL_VERSION            = 2.0               # See which protocol version is used in the Dynamixel

# Default setting
BAUDRATE                    = 1000000             # Dynamixel default baudrate : 57600
DEVICENAME                  = '/dev/ttyDXL_gello'    # Check which port is being used on your controller

portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Open port
if portHandler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    quit()

# Set port baudrate
if portHandler.setBaudRate(BAUDRATE):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    quit()

print("Scanning for Dynamixels...")
# Scan for Dynamixels
for id in range(1, 253):
    model_number, dxl_comm_result, dxl_error = packetHandler.ping(portHandler, id)
    if dxl_comm_result == COMM_SUCCESS:
        print(f"[ID:{id:03d}] ping succeeded. Model Number: {model_number}")
    # else:
    #     print(f"[ID:{id:03d}] ping failed.")

portHandler.closePort()
