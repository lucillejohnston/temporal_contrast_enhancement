# Script for running the TCE (OH/OA) Task with QST 
"""
This script runs the offset analgesia (OA) and offset hyperalgesia (OH) task with QST

It extracts calibration data, so be sure to run TCE_calibration.py beforehand 

BEFORE RUNNING
- Make sure to set the correct COM port for the thermode in the YAML file
- Make sure to set the correct number of blocks in the generate_trial_order function
 
Author: Lucille Johnston
Updated: 5/16/25
- upated to use new eVAS slider
"""

# Import necessary libraries
from datetime import datetime
import TcsControl_py3 as tcs 
# import VAS_slider as vas
import eVAS_slider as evas 
import tkinter as tk # for GUI
from tkinter import messagebox
import yaml, time, threading, random, csv, os

# ########### FOR TROUBLESHOOTING ONLY ##########
# import virtual_QST as tcs
# thermode = tcs.VirtualThermode()
# com_port = 'virtual'
# ###############################################

# ----------------------------------
#  Initial Patient Info GUI 
# ----------------------------------

class PatientInfoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Enter Patient Information")
        self.root.geometry("400x500")
        self.label_name = tk.Label(root, text='Patient Name:', font=('Arial', 20, 'bold'))
        self.label_name.pack(pady=20, padx=10)
        self.entry_name = tk.Entry(root, font=('Arial', 20))
        self.entry_name.pack(pady=20, padx=10)
        self.label_trial_start = tk.Label(root, text="Trial Start On (if not 1):", font=('Arial', 20, 'bold'))
        self.label_trial_start.pack(pady=10, padx=10)
        self.entry_trial_start = tk.Entry(root, font=('Arial', 20))
        self.entry_trial_start.pack(pady=10, padx=10)
        self.submit_button = tk.Button(root, text="Submit", font=('Arial',20), command=self.submit)
        self.submit_button.pack(pady=20, padx=20)
        self.patient_info = None
        self.T1 = None
        self.T2 = None
        self.com_port = None
        self.num_blocks = None
        self.trial_start = None

    def submit(self):
        name = self.entry_name.get().strip()
        trial_start = self.entry_trial_start.get().strip()
        if not name:
            messagebox.showerror("Input Error", "Please enter a valid name")
            return
        date_str = datetime.now().strftime("%Y%m%d")
        if not trial_start:
            self.trial_start = 1
        else:
            try:
                self.trial_start = int(trial_start)
            except ValueError:
                messagebox.showerror("Input Error", "Trial start must be a number")
                return
        self.patient_info = {"name": name,
                             "date": date_str}
        config_dir = os.path.join(os.path.dirname(__file__), "configs")
        file_name = f"{self.patient_info['name']}_{self.patient_info['date']}_config.yaml"
        full_path = os.path.join(config_dir,file_name)
        config = self.load_config_file(full_path)
        if config:
            self.side = config.get("side",None)
            self.T1 = config.get("T1",None)
            self.T2 = self.T1 + 1
            self.com_port = config.get("COM_Port",None)
            self.num_blocks = config.get("num_blocks",None)
            if self.T1 is None:
                messagebox.showerror("Calibration Data Error", "T1 value is missing in the config file")
                return  
            if self.com_port is None:
                messagebox.showerror("Calibration Data Error","COM Port is missing in the config file")
                return
            messagebox.showinfo("Calibration Loaded",f"T1 = {self.T1}; T2 = {self.T2}")
        else:
            messagebox.showerror("File Not Found",f"No calibration file found for {name} on {date_str}")
            return
        self.root.destroy()
    
    def load_config_file(self, file_name):
        try: 
            with open(file_name,'r') as file:
                config = yaml.safe_load(file)
            return config
        except yaml.YAMLError:
            print(f"Error: The file {file_name} is not a valid YAML file.")
            return None

def get_patient_calibration():
    root = tk.Tk()
    app = PatientInfoGUI(root)
    root.mainloop()
    return app.T1, app.T2, app.com_port, app.num_blocks, app.side, app.trial_start, app.patient_info
    
# ---------------------------------
# Define Task Parameters
# ---------------------------------
baselineTemp = 29 # resting temperature in °C
ramp_speed = 1.5 # °C/s
return_speed = 6 # °C/s
# ---------------------------------
target_generator = tcs.target_zone_generator()  # Create a target zone generator

def generate_trial_order(num_blocks):
    trial_types = ['hold_T1','step_up','step_down','hold_T2']
    order = []
    for _ in range(num_blocks):
        block = trial_types[:] # make a copy
        random.shuffle(block)
        order.extend(block)
    return order

def update_config(config_file, num_blocks):
    trial_order_list = generate_trial_order(num_blocks)
    trial_order_mapping = {i+1: trial for i, trial in enumerate(trial_order_list)}
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    config['trial_order'] = trial_order_mapping
    with open(config_file, 'w') as file:
        yaml.dump(config, file, indent=4)
    return trial_order_mapping
# ---------------------------------
class ExperimenterGUI:
    def __init__(self, root, thermode, T1, T2, patient_info, side, trial_start, trial_order_mapping, baseline, ramp_speed, return_speed):
        self.thermode = thermode
        self.T1 = T1
        self.T2 = T2
        self.patient_info = patient_info
        self.vas_slider = vas.VASSlider()
        # Build the GUI
        self.root = root
        self.root.geometry("1200x1000")
        self.root.title("TCE Task: Experimenter Interface")
        self.log_text = tk.Text(self.root, height=20, width=60, font=('Arial',16))
        self.log_text.pack(pady=10)
        self.trial_label = tk.Label(self.root, text="Trial Number: 0", font=('Arial',20))
        self.trial_label.pack(pady=10)

        self.notes_label = tk.Label(self.root, text="Notes:", font=('Arial', 16))
        self.notes_label.pack(pady=10)
        self.notes_entry = tk.Entry(self.root, font=('Arial', 16))
        self.notes_entry.pack(pady=10)
        self.notes_entry.bind("<Return>", self.add_note)

        self.trial_start = trial_start
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        self.next_trial_button = tk.Button(btn_frame, text="Next Trial", font=('Arial',16), command=self.start_trial)
        self.next_trial_button.grid(row=0, column=0, padx=10, pady=10)

        self.repeat_trial_button = tk.Button(btn_frame, text="Repeat Last Trial", font=('Arial',16), command=self.repeat_last_trial)
        self.repeat_trial_button.grid(row=0, column=1, padx=10, pady=10)
        
        self.stop_button = tk.Button(btn_frame, text="Stop", font = ('Arial',16), command=self.stop_trial)
        self.stop_button.grid(row=0, column=2, padx=10, pady=10)

        # Lists to store data
        self.trial_samples = [] #store continuous samples (trial_count, trial_type, timestamp, pain_rating, temp)
        self.notes = []
        self.data = [] # List of dictionaries, one per trial 
        self.sampling_interval = 100 # 10Hz = every 100ms 
        self.recording = False
        self.trial_in_progress = False
        self.trial_completed = False
        self.trial_start_time = None 
        self.cancelled = False

        # Parameters
        self.baseline = baseline
        self.ramp_speed = ramp_speed
        self.return_speed = return_speed
        self.target_gen = target_generator 
        self.side = side
        self.trial_order_mapping = trial_order_mapping
        self.trials = list(self.trial_order_mapping.values())
        self.trial_number = 0 # trial number starts at 0 for the session

    def log(self, message): # append a message to log 
        def append():
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
        self.root.after(0, append)

    def add_note(self, event=None):
        note = self.notes_entry.get().strip()
        if note:
            timestamp = self.get_current_time(time.time())
            full_note = f"{timestamp}: {note}"
            self.log(full_note)
            self.notes.append((timestamp, note))
            self.notes_entry.delete(0, tk.END)

    def start_sampling(self):
        self.log("Starting continuous sampling")
        self.sampling_active = True
        self.sample_loop()
    
    def get_current_time(self, current_time):
        current_time_str = time.strftime("%H:%M:%S", time.localtime(current_time))
        milliseconds = int((current_time - int(current_time)) * 1000)
        return f"{current_time_str}.{milliseconds:03d}"
    
    def sample_loop(self):
        if not self.sampling_active:
            return
        current_time = time.time()
        current_time_str = self.get_current_time(current_time)
        vas = self.vas_slider.get_value(self.side)
        temp = self.thermode.get_temperatures()
        if temp and isinstance(temp, list) and all(isinstance(t, float) for t in temp):
            self.trial_samples.append((self.trial_number, self.trial_type, current_time_str, vas, temp))
        self.root.after(self.sampling_interval, self.sample_loop)

    def update_trial_label(self):
        def update():
            self.trial_label.config(text=f"Trial: {self.trial_number}")
        self.root.after(0, update)

    def start_trial(self):
        if self.cancelled:
            return
        if self.trial_in_progress:
            return
        if self.trial_number == 0 and self.trial_start > 1: 
            self.trial_number = self.trial_start - 1
        
        if self.trial_number >= len(self.trials):
            self.log("All trials completed")
            self.finish_session()
            return
        self.trial_type = self.trials[self.trial_number]
        active_zones = next(target_generator)

        self.trial_number += 1
        self.update_trial_label()
        self.log(f"Starting Trial {self.trial_number}: {self.trial_type} ")
        self.trial_in_progress = True
        self.start_sampling()

        # Disable buttons while the trial runs
        self.next_trial_button.config(state='disabled')
        self.repeat_trial_button.config(state='disabled')
        self.stop_button.config(state='normal')

        # Run the trial in the background so the GUI remains responsive
        trial_thread = threading.Thread(target=self.run_trial_by_type, args=(self.trial_type,active_zones))
        trial_thread.daemon = True
        trial_thread.start()

    def repeat_last_trial(self):
        if self.trial_number == 0:
            messagebox.showerror("Error", "No trial to repeat")
            return
        self.log(f"Repeating Trial {self.trial_number}")
        self.trial_number -= 1
        self.trial_type = self.trials[self.trial_number]
        active_zones = next(target_generator)
        # Disable all the buttons
        self.next_trial_button.config(state=tk.DISABLED)
        self.repeat_trial_button.config(state=tk.DISABLED)
        self.finish_button.config(state=tk.DISABLED)
        # Return to the same trial
        trial_thread = threading.Thread(target=self.run_trial_by_type, args=(self.trial_type, active_zones))
        trial_thread.daemon = True
        trial_thread.start()

    def stop_trial(self):
        if self.trial_number == 0:
            messagebox.showwarning("Warning","No trial has started yet.")
            self.root.destroy()
        elif self.trial_number % 4 != 0:
            trials_in_block = 4 - (self.trial_number % 4)
            proceed = messagebox.askyesno("Warning",f"Mid-block! Complete {trials_in_block} more trials to finish this block?")
            if proceed:
                return
            else:
                self.log("Stopping session...")
                self.cancelled = True
                self.sampling_active = False
                self.save_data('TCE_data_stopped')
        else:
            proceed = messagebox.askyesno("Warning", "Are you sure you want to stop the session?")
            if proceed:
                self.log("Stopping session...")
                self.cancelled = True
                self.sampling_active = False
                self.save_data('TCE_data_stopped')
            else:
                return
    
    def run_trial_by_type(self, trial_type, active_zones):
        self.log(f"Active zones: {active_zones}")
        if trial_type == 'hold_T1':
            self.run_trial_hold_T1(active_zones)
        elif trial_type == 'step_up':
            self.run_trial_step_up(active_zones)
        elif trial_type == 'step_down':
            self.run_trial_step_down(active_zones)
        elif trial_type == 'hold_T2':
            self.run_trial_hold_T2(active_zones)
        if not self.cancelled:
            self.log(f"Trial {self.trial_number} complete")
            self.log("Wait 20s between trials...")
            time.sleep(20) # wait for 20s then end trial 
            self.sampling_active = False
            self.log("Stopping continuous sampling")
            self.trial_in_progress = False
            self.end_trial()
    
    def run_trial_hold_T1(self, active_zones):
        # Ramp up from baseline to T1
        ramp_duration = (self.T1 - self.baseline) / self.ramp_speed
        self.thermode.set_durations([ramp_duration if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T1 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Ramping up to T1 ({self.T1}) over {ramp_duration:.2f}s")
        time.sleep(ramp_duration)
        # Hold at T1 for A+B+C duration
        hold_duration = 30
        self.thermode.set_durations([hold_duration if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T1 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Holding at T1 ({self.T1}) for {hold_duration:.2f}s")
        time.sleep(hold_duration)
        # Ramp down to baseline
        ramp_down_duration = (self.T1 - self.baseline) / self.return_speed
        self.thermode.set_durations([ramp_down_duration if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.baseline if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Ramping down to baseline ({self.baseline}) over {ramp_down_duration:.2f}s")
        time.sleep(ramp_down_duration)

    def run_trial_step_up(self, active_zones):
        # Ramp up from baseline to T1
        ramp_duration_A = (self.T1 - self.baseline) / self.ramp_speed
        self.thermode.set_durations([ramp_duration_A if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T1 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Ramping up to T1 ({self.T1}) over {ramp_duration_A:.2f}s")
        time.sleep(ramp_duration_A)
        # Hold at T1 for A duration
        hold_duration_A = 5
        self.thermode.set_durations([hold_duration_A if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T1 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Holding at T1 ({self.T1}) for {hold_duration_A:.2f}s")
        time.sleep(hold_duration_A)
        # Ramp up to T2
        ramp_duration_step = 1 / self.ramp_speed
        self.thermode.set_durations([ramp_duration_step if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T2 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Ramping up to T2 ({self.T2}) over {ramp_duration_step:.2f}s")
        time.sleep(ramp_duration_step)
        # Hold at T2 for B duration
        hold_duration_B = 5
        self.thermode.set_durations([hold_duration_B if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T2 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Holding at T2 ({self.T2}) for {hold_duration_B:.2f}s")
        time.sleep(hold_duration_B)
        # Ramp down to T1
        self.thermode.set_durations([ramp_duration_step if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T1 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Ramping down to T1 ({self.T1}) over {ramp_duration_step:.2f}s")
        time.sleep(ramp_duration_step)
        # Hold at T1 for C duration
        hold_duration_C = 20
        self.thermode.set_durations([hold_duration_C if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T1 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Holding at T1 ({self.T1}) for {hold_duration_C:.2f}s")
        time.sleep(hold_duration_C)
        # Ramp down to baseline
        ramp_down_duration = (self.T1 - self.baseline) / self.return_speed
        self.thermode.set_durations([ramp_down_duration if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.baseline if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Ramping down to baseline ({self.baseline}) over {ramp_down_duration:.2f}s")
        time.sleep(ramp_down_duration)
    
    def run_trial_step_down(self, active_zones):
        # Ramp up from baseline to T2
        ramp_duration_A = (self.T2 - self.baseline) / self.ramp_speed
        self.thermode.set_durations([ramp_duration_A if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T2 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Ramping up to T2 ({self.T2}) over {ramp_duration_A:.2f}s")
        time.sleep(ramp_duration_A)
        # Hold at T2 for A duration
        hold_duration_A = 5
        self.thermode.set_durations([hold_duration_A if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T2 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Holding at T2 ({self.T2}) for {hold_duration_A:.2f}s")
        time.sleep(hold_duration_A)
        # Ramp down to T1
        ramp_duration_step = 1 / self.ramp_speed
        self.thermode.set_durations([ramp_duration_step if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T1 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Ramping down to T1 ({self.T1}) over {ramp_duration_step:.2f}s")
        time.sleep(ramp_duration_step)
        # Hold at T1 for B duration
        hold_duration_B = 5
        self.thermode.set_durations([hold_duration_B if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T1 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Holding at T1 ({self.T1}) for {hold_duration_B:.2f}s")
        time.sleep(hold_duration_B)
        # Ramp up to T2
        self.thermode.set_durations([ramp_duration_step if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T2 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Ramping up to T2 ({self.T2}) over {ramp_duration_step:.2f}s")
        time.sleep(ramp_duration_step)
        # Hold at T2 for C duration
        hold_duration_C = 20
        self.thermode.set_durations([hold_duration_C if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T2 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Holding at T2 ({self.T2}) for {hold_duration_C:.2f}s")
        time.sleep(hold_duration_C)
        # Ramp down to baseline
        ramp_down_duration = (self.T2 - self.baseline) / self.return_speed
        self.thermode.set_durations([ramp_down_duration if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.baseline if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Ramping down to baseline ({self.baseline}) over {ramp_down_duration:.2f}s")
        time.sleep(ramp_down_duration)
    
    def run_trial_hold_T2(self, active_zones):
        # Ramp up from baseline to T2
        ramp_duration = (self.T2 - self.baseline) / self.ramp_speed
        self.thermode.set_durations([ramp_duration if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T2 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Ramping up to T2 ({self.T2}) over {ramp_duration:.2f}s")
        time.sleep(ramp_duration)
        # Hold at T2 for A+B+C duration
        hold_duration = 30
        self.thermode.set_durations([hold_duration if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T2 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Holding at T2 ({self.T2}) for {hold_duration:.2f}s")
        time.sleep(hold_duration)
        # Ramp down to baseline
        ramp_down_duration = (self.T2 - self.baseline) / self.return_speed
        self.thermode.set_durations([ramp_down_duration if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.baseline if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Ramping down to baseline ({self.baseline}) over {ramp_down_duration:.2f}s")
        time.sleep(ramp_down_duration)
   
    def end_trial(self):
        self.trial_completed = True
        data_dir = os.path.join(os.path.dirname(__file__), "task_data")
        file_name = f"{self.patient_info['name']}_{self.patient_info['date']}_temp_taskdata.csv"
        full_path = os.path.join(data_dir,file_name)
        file_exists = os.path.exists(full_path)
        try:
            with open(full_path, 'a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(["trial_count","trial_type","timestamp", "pain_rating", "temperature"])
                for sample in self.trial_samples:
                    writer.writerow(sample)
                writer.writerow([]) # add blank line
                writer.writerow(["timestamp","note"])
                for note in self.notes:
                    writer.writerow(note)
                self.log(f"Trial {self.trial_number} data saved to {file_name}")
                self.trial_samples.clear()
                self.notes.clear()
        except Exception as e:
            self.log(f"Error saving trial data: {e}")
        if self.trial_number == len(self.trials):
            self.log("All trials completed")
            self.finish_session()
            return
        else:
            self.enable_buttons () # Re-enable buttons for next trial
    
    def enable_buttons(self): # Re-enable buttons
        self.next_trial_button.config(state='normal')
        self.repeat_trial_button.config(state='normal')
        self.stop_button.config(state='normal')
        self.log("Ready to start next trial")

    def finish_session(self):
        self.sampling_active = False
        self.log("Finishing sampling data")
        self.save_data('TCE_data')
        self.root.destroy()
    
    def save_data(self, filename_suffix):
        data_dir = os.path.join(os.path.dirname(__file__), "task_data")
        base_name = f"{self.patient_info['name']}_{self.patient_info['date']}_{filename_suffix}"
        full_path = os.path.join(data_dir,base_name)
        filename = f"{full_path}.csv"
        n=2
        while os.path.exists(filename):
            if not messagebox.askyesno("Overwrite File", f"File {filename} already exists. Overwrite?"):
                filename = f"{full_path}_{n}.csv"
                n+=1
            else:
                break
        try:
            temp_file_path = os.path.join(data_dir, f"{self.patient_info['name']}_{self.patient_info['date']}_temp_taskdata.csv")
            if os.path.exists(temp_file_path): 
                with open(temp_file_path, 'r') as temp_file:
                    data = temp_file.read()
                with open(filename, 'w', newline='') as dst:
                    dst.write(data)
            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([])  # add blank line to separate sections
                writer.writerow(['Experimenter Log'])
                log_content = self.log_text.get("1.0", tk.END).splitlines()
                for line in log_content:
                    writer.writerow([line])
            self.log(f"Session data saved to {base_name}")
            messagebox.showinfo("Session Complete",f"Session data saved to {base_name}")
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            self.root.destroy()
        except Exception as e:
            messagebox.showerror("File Error", f"Could not save data: {e}")

    def run(self):
        self.root.mainloop()

# -------------------------------------
# Main Application
# -------------------------------------
def main():
    T1, T2, com_port, num_blocks, side, trial_start, patient_info = get_patient_calibration()
    if T1 is None:
        exit("Calibration data missing. Exiting.")
    thermode = tcs.TcsDevice(port=com_port)
    thermode.set_quiet()  # set device to quiet mode - shouldn't heat up between things
    thermode.set_baseline(baselineTemp)
    thermode.set_ramp_speed([ramp_speed]*5)
    thermode.set_return_speed([return_speed]*5)

    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    config_file = os.path.join(config_dir, f"{patient_info['name']}_{patient_info['date']}_config.yaml")
    trial_order_mapping = update_config(config_file, num_blocks)

    root = tk.Tk()
    experimenter_gui = ExperimenterGUI(root, thermode, T1, T2, patient_info, side, 
                                       trial_start=trial_start, 
                                       trial_order_mapping=trial_order_mapping,
                                       baseline=baselineTemp, 
                                       ramp_speed=ramp_speed,
                                       return_speed=return_speed)
    experimenter_gui.run()

if __name__ == "__main__":
    main()
