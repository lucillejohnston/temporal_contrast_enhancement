"""
Script to send TTL pulses via serial port and log timestamps.
Works with TCE_calibration and TCE_task 

TTL Codes:
- 5 pulses indicate start and end of task
- 3 pulses indicate start and end of trial
- 1 pulse indicates start and end of hold phase (within a trial)
"""
import serial, os, time, sys
from datetime import datetime

def open_serial_port(port_name="/dev/tty.usbserial-BBTKUSBTTL", baud_rate=115200):
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

def get_patient_info(default_save_folder, session='task'):
    """
    Prompts for patient ID, creates a log folder, and returns the log file path.
    
    Parameters:
        default_save_folder (str): The root folder for saving logs.
        session (str): Indicates the session type. Accepts 'calibration' or 'task'.
                       Defaults to 'task'.
    """
    # Set folder based on session type.
    session_folder = "calibration_data" if session.lower() == "calibration" else "task_data"
    
    patient_id = input("Enter patient ID: ").strip()
    timestamp_str = datetime.now().strftime("%y%m%d")
    filename = f"{patient_id}_{timestamp_str}_TTLtimestamps.csv"

    dest_folder = os.path.join(default_save_folder, session_folder)
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

def send_session_start_ttl(ser, logfile, pulse_pin=2, pulse_width=0.1, inter_pulse_delay=0.05):
    """
    Sends 5 TTL pulses for the start and end of a session.
    """
    for i in range(5):
        send_ttl_pulse(ser, logfile, pulse_pin, pulse_width)
        time.sleep(inter_pulse_delay)

def send_session_end_ttl(ser, logfile, pulse_pin=2, pulse_width=0.1, inter_pulse_delay=0.05):
    """
    Sends 7 TTL pulses for the end of a session.
    """
    for i in range(7):
        send_ttl_pulse(ser, logfile, pulse_pin, pulse_width)
        time.sleep(inter_pulse_delay)

def send_trial_ttl(ser, logfile, pulse_pin=2, pulse_width=0.1, inter_pulse_delay=0.05):
    """
    Sends 3 TTL pulses for the start and end of a trial.
    """
    for i in range(3):
        send_ttl_pulse(ser, logfile, pulse_pin, pulse_width)
        time.sleep(inter_pulse_delay)

def send_event_ttl(ser, logfile, pulse_pin=2, pulse_width=0.1, inter_pulse_delay=0.05):
    """
    Sends 1 TTL pulse for the start and end of a hold phase.
    """
    send_ttl_pulse(ser, logfile, pulse_pin, pulse_width)
    time.sleep(inter_pulse_delay)

if __name__ == "__main__":
    # Set the file save location
    save_folder = os.path.join(os.path.dirname(__file__), "TTL_data")

    # Open the serial port.
    port_name = "/dev/tty.usbserial-BBTKUSBTTL"
    ser = open_serial_port(port_name)

    # send_ttl_pulse(ser, sys.stdout, pulse_pin=2, pulse_width=0.1)
    # print("Test pulse sent")
    # ser.close()
    
    # Get patient info and determine log file path.
    log_file_path = get_patient_info(save_folder, session='task')
    
    # Open the log file in append mode and send the TTL pulse.
    with open(log_file_path, "a") as logfile:
        send_ttl_pulse(ser, logfile, pulse_pin=2, pulse_width=0.1)
    
    print(f"File saved to: {log_file_path}")

    # Close the serial port.
    ser.close()
    print("Serial port closed.")
