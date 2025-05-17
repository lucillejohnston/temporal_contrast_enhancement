# Script for running the calibration before TCE task
"""
This script is for introducing the TCE task on Demo Day 
Author: Lucille Johnston
Updated: 05/08/25

Phase 1: Learn how to use the COVAS slider 
- follow the slider movement to the display 
- after staying within a threshold, proceed to the next step

Phase 2: Learn how to follow temperature with COVAS slider
- follow the slider movement to the temperature
- after staying within a threshold, proceed to the next step

Phase 3: A few test TCE trials
- crucially, have participant follow slider to SENSATION, rather than pain 
- a few offset, onset, and hold trials 
- account for learning how the task works
"""
import tkinter as tk # for GUI
from tkinter import messagebox
from datetime import datetime
import os, sys, yaml, random
import subprocess
import TcsControl_py3 as tcs

com_port = '/dev/tty.usbmodem1401' ########## CHANGE TO MATCH DEVICE ##########
"""
On windows: Task bar > USB hardware + media > open devices + printers > MCP2221
- Double click MCP2221 to open properties > hardware tab --> should have MCP2221 (COM ##) 
example: com_port = COM3

On Mac: Terminal > ls /dev/tty.*
- Should have something like /dev/tty.usbmodem11101
example: com_port = /dev/tty.usbmodem1401
"""

# --- Initial Patient Info GUI ---
class PatientInfoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Enter Patient Information")
        self.root.geometry("700x500")
        self.label_name = tk.Label(root, text='Patient Name:', font=('Arial', 20, 'bold'))
        self.label_name.pack(pady=20, padx=10)
        self.entry_name = tk.Entry(root, font=('Arial', 20))
        self.entry_name.pack(pady=20, padx=10)
        self.label_side = tk.Label(root, text='Which side is the thermode on?', font=('Arial', 20, 'bold'))
        self.label_side.pack(pady=20, padx=10)
        self.side_var = tk.StringVar(value="left")
        self.radio_left = tk.Radiobutton(root, text="Left", variable=self.side_var, value="left", font=('Arial', 20))
        self.radio_left.pack(pady=10)
        self.radio_right = tk.Radiobutton(root, text="Right", variable=self.side_var, value="right", font=('Arial', 20))
        self.radio_right.pack(pady=10)

        self.submit_button = tk.Button(root, text="Submit", font=('Arial',20), command=self.submit)
        self.submit_button.pack(pady=20, padx=20)

        self.patient_info = None

    def submit(self):
        name = self.entry_name.get().strip()
        if not name:
            messagebox.showerror("Input Error", "Please enter a valid name")
            return
        date_str = datetime.now().strftime("%Y%m%d")
        self.patient_info = {"name": name, "date": date_str}
        self.side = self.side_var.get()
        self.root.destroy()

def get_patient_info():
    # Create a temporary Tk root for the patient info GUI
    info_root = tk.Tk()
    patient_info_gui = PatientInfoGUI(info_root)
    # Start the GUI, which will call quit() and destroy itself in submit()
    info_root.mainloop()
    # Ensure the GUI is fully closed even if it wasn’t already
    try:
        info_root.destroy()
    except Exception:
        pass
    # Retrieve patient info 
    patient_info = patient_info_gui.patient_info
    side = patient_info_gui.side
    if not patient_info:
        exit("No patient info provided.")
    return patient_info, side

def make_config(patient_info, side):
    config_dir = os.path.join(os.path.dirname(__file__), "demo_day_data")
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    config_file = os.path.join(config_dir, f"{patient_info['name']}_{patient_info['date']}_config.yaml")
    config = {
        "COM_port": com_port,
        "T1": 43, # just picked a temperature that shouldn't be too painful, hard coded for now not calibrated
        "patient_info": patient_info,
        "side": side,
        "trial_order": { # hard coded for now
            1: "hold_T2",
            2: "step_down",
            3: "step_up",
            4: "hold_T1"
        }
    }
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    print(f"Configuration file saved to {config_file}")
    return config_file

# --- PHASE 1 Practice Slider ---
def run_slider_practice():
    script_path = os.path.join(os.path.dirname(__file__), "slider_practice.py")
    if not os.path.exists(script_path):
        print(f"slider_practice.py not found at {script_path}")
        return
    try:
        # Run asynchronously so the main GUI isn’t blocked.
        process = subprocess.Popen([sys.executable, script_path])
        # Optionally wait until the slider script completes before proceeding:
        process.wait()
    except Exception as e:
        print(f"Error running slider_practice.py: {e}")


# --- PHASE 2 PATIENT GUI (Ramp + Hold Calibration) ---
def run_temp_seesaw(config_file):
    script_path = os.path.join(os.path.dirname(__file__), "temp_seesaw.py")
    if not os.path.exists(script_path):
        print(f"temp_seesaw.py not found at {script_path}")
        return
    try:
        # Run asynchronously so the main GUI isn’t blocked.
        process = subprocess.Popen([sys.executable, script_path, config_file])
        # Optionally wait until the seesaw script completes before proceeding:
        process.wait()
    except Exception as e:
        print(f"Error running temp_seesaw.py: {e}")

# --- PHASE 3 A FEW TCE TRIALS ---
def run_tce_trials(config_file):
    script_path = os.path.join(os.path.dirname(__file__), "TCE_task_demo.py")
    if not os.path.exists(script_path):
        print(f"TCE_task.py not found at {script_path}")
        return
    try:
        # Run asynchronously so the main GUI isn’t blocked.
        process = subprocess.Popen([sys.executable, script_path, config_file])
        # Optionally wait until the TCE trials script completes before proceeding:
        process.wait()
    except Exception as e:
        print(f"Error running TCE_task.py: {e}")

# --- Main ---
if __name__ == '__main__':
    thermode = tcs.TcsDevice(port=com_port)
    thermode.set_quiet()  # set device to quiet mode - shouldn't heat up between things
    # Get patient info
    patient_info, side = get_patient_info()
    config_file = make_config(patient_info, side)

    # PHASE 1
    run_slider_practice()

    # PHASE 2
    run_temp_seesaw(config_file)

    # PHASE 3
    run_tce_trials(config_file)
