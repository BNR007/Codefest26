import serial
import time

ser = serial.Serial('COM7', 9600, timeout=1)
time.sleep(2)

motor_position = 'center'

def move_left():
    global motor_position
    motor_position = 'left'
  
def move_right():
    global motor_position
    motor_position = 'right'
  
def move_center():
    global motor_position
    motor_position = 'center'
    
while True:
    line = ser.readline().decode(errors='ignore').strip()
    if line:
        print(line)
move_right()
time.sleep(2)