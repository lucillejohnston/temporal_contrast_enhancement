"""
Script to send TTL pulses via serial port and log timestamps.

Fully written by ChatGPT

Untested

"""
import serial
import os
import time
from datetime import datetime

def open_serial_port(port_name="/dev/tty.usbserial", baud_rate=115200):
    """
    Opens the serial port with the given parameters.
    Adjust port_name to suit your hardware.
    Use ls /dev/tty.* to find the correct port on macOS.
    On Windows, go to Device Manager and find the COM port.
    """
    try:
        ser = serial.Serial(
            port=port_name,
            baudrate=baud_rate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )
        # Clear buffers and send initial command 'RR'
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        ser.write(b'RR')
        print("Port opened and command 'RR' sent.")
        time.sleep(1)
        return ser
    except Exception as e:
        print("Error opening serial port:", e)
        raise

def get_patient_info(default_save_folder):
    """
    Prompts for patient ID, creates a log folder, and returns the log file path.
    """
    patient_id = input("Enter patient ID: ").strip()
    timestamp_str = datetime.now().strftime("%y%m%d-%H%M")
    filename = f"{patient_id}_{timestamp_str}_TTLtimestamps.csv"

    # Create folder: default_save_folder/PatientID/TTLtimestamps
    dest_folder = os.path.join(default_save_folder, patient_id, "TTLtimestamps")
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    print(f"Patient ID is {patient_id}")
    print(f"Saving files to {dest_folder}")
    return os.path.join(dest_folder, filename)

def send_ttl_pulse(ser, logfile, pulse_pin=2, pulse_width=0.1):
    """
    Sends a TTL pulse via the serial connection and logs the timestamp.
    """
    try:
        # Convert pulse_pin to two-digit hexadecimal string.
        pulse_on = f"{pulse_pin:02X}"
        pulse_off = f"{0:02X}"
        
        # Send the TTL pulse.
        ser.write(pulse_on.encode())
        time.sleep(pulse_width)
        ser.write(pulse_off.encode())
        
        # Log the current timestamp.
        pulse_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
        logfile.write(pulse_time + "\n")
        logfile.flush()
        print(f"Pulse delivered: {pulse_time}")
    except Exception as e:
        print("Error during TTL pulse:", e)
        raise

if __name__ == "__main__":
    # Set the file save location (update this path if necessary)
    save_folder = "/Users/paulettebogan/UCSF DBS for Pain Dropbox/PainNeuromodulationLab/DATA ANALYSIS/Victoria"

    # Open the serial port.
    # For macOS, update the port name as needed (e.g., /dev/tty.usbmodemXXXX)
    port_name = "/dev/tty.usbserial"
    ser = open_serial_port(port_name)
    
    # Get patient info and determine log file path.
    log_file_path = get_patient_info(save_folder)
    
    # Open the log file in append mode and send the TTL pulse.
    with open(log_file_path, "a") as logfile:
        send_ttl_pulse(ser, logfile, pulse_pin=2, pulse_width=0.1)
    
    print(f"File saved to: {log_file_path}")
    
    # Close the serial port.
    ser.close()
    print("Serial port closed.")
    
    # Troubleshooting: Uncomment below to send a test pulse.
    """
    with open(log_file_path, "a") as logfile:
        send_ttl_pulse(ser, logfile, pulse_pin=2, pulse_width=0.1)
    """