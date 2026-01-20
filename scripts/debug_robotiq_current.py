
import time
import socket
import sys
import threading
from collections import OrderedDict

# Minimal Robotiq Client to debug COU
class RobotiqDebugger:
    def __init__(self, ip, port=63352):
        self.ip = ip
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(2.0)
        
    def connect(self):
        print(f"Connecting to {self.ip}:{self.port}...")
        self.socket.connect((self.ip, self.port))
        print("Connected.")

    def get_var(self, name):
        cmd = f"GET {name}\n"
        self.socket.sendall(cmd.encode("UTF-8"))
        data = self.socket.recv(1024).decode("UTF-8").strip()
        # Expected: "VAR x"
        parts = data.split()
        if len(parts) >= 2 and parts[0] == name:
            return int(parts[1])
        return f"ERR({data})"

    def run(self):
        print("Reading gripper variables... Press Ctrl+C to stop.")
        print(f"{'POS':>5} | {'OBJ':>5} | {'COU (Current)':>15} | {'STA':>5} | {'FLT':>5}")
        print("-" * 50)
        
        try:
            while True:
                pos = self.get_var("POS")
                obj = self.get_var("OBJ")
                cou = self.get_var("COU")
                sta = self.get_var("STA")
                flt = self.get_var("FLT")
                
                print(f"{pos:>5} | {obj:>5} | {cou:>15} | {sta:>5} | {flt:>5}")
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            self.socket.close()

if __name__ == "__main__":
    # Default IP from your config
    IP = "192.168.1.11" 
    PORT = 63352
    
    debugger = RobotiqDebugger(IP, PORT)
    try:
        debugger.connect()
        debugger.run()
    except Exception as e:
        print(f"\nError: {e}")
