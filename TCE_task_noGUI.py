"""
This script runs the TCE (OH/OA) task via the terminal without a GUI.
It prints log messages to the terminal and uses input prompts for interactions.


"""
import time, threading, random, csv, os
import TcsControl_py3 as tcs 
import VAS_slider as vas
import yaml
import tkinter as tk
from tkinter import messagebox
from datetime import datetime


# Parameters
baselineTemp = 29  # resting temperature in °C
ramp_speed = 1.5   # °C/s
return_speed = 6   # °C/s

target_generator = tcs.target_zone_generator()  # Create a target zone generator

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
    
def generate_trial_order(num_blocks):
    trial_types = ['hold_T1', 'step_up', 'step_down', 'hold_T2']
    order = []
    for _ in range(num_blocks):
        block = trial_types[:]  # make a copy
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

class ExperimenterCLI:
    def __init__(self, thermode, T1, T2, patient_info, side, trial_start, trial_order_mapping, baseline, ramp_speed, return_speed):
        self.thermode = thermode
        self.T1 = T1
        self.T2 = T2
        self.patient_info = patient_info
        self.vas_slider = vas.VASSlider()
        self.trial_start = trial_start
        self.trial_samples = []  # store continuous samples (trial_count, trial_type, timestamp, pain_rating, temp)
        self.notes = []
        self.sampling_interval = 0.1  # 10Hz
        self.sampling_active = False
        self.trial_in_progress = False
        self.cancelled = False
        self.baseline = baseline
        self.ramp_speed = ramp_speed
        self.return_speed = return_speed
        self.side = side
        self.trial_order_mapping = trial_order_mapping
        self.trials = list(self.trial_order_mapping.values())
        self.trial_number = 0  # trial count starts at 0

    def log(self, message):
        print(message)

    def add_note(self):
        note = input("Enter note (or leave blank): ").strip()
        if note:
            timestamp = self.get_current_time(time.time())
            full_note = f"{timestamp}: {note}"
            self.log(full_note)
            self.notes.append((timestamp, note))

    def get_current_time(self, current_time):
        current_time_str = time.strftime("%H:%M:%S", time.localtime(current_time))
        milliseconds = int((current_time - int(current_time)) * 1000)
        return f"{current_time_str}.{milliseconds:03d}"
    
    def sample_loop(self):
        while self.sampling_active:
            current_time = time.time()
            current_time_str = self.get_current_time(current_time)
            vas_value = self.vas_slider.get_value(self.side)
            temp = self.thermode.get_temperatures()
            if temp and isinstance(temp, list) and all(isinstance(t, float) for t in temp):
                # Save sample with current trial (if defined)
                self.trial_samples.append((self.trial_number, self.trial_type, current_time_str, vas_value, temp))
            time.sleep(self.sampling_interval)

    def start_sampling(self):
        self.log("Starting continuous sampling")
        self.sampling_active = True
        self.sampling_thread = threading.Thread(target=self.sample_loop)
        self.sampling_thread.daemon = True
        self.sampling_thread.start()

    def start_trial(self):
        if self.cancelled or self.trial_in_progress:
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
        self.log(f"Starting Trial {self.trial_number}: {self.trial_type}")
        self.trial_in_progress = True
        self.start_sampling()
        # Run trial in separate thread
        trial_thread = threading.Thread(target=self.run_trial_by_type, args=(self.trial_type, active_zones))
        trial_thread.daemon = True
        trial_thread.start()

    def repeat_last_trial(self):
        if self.trial_number == 0:
            self.log("Error: No trial to repeat")
            return
        self.log(f"Repeating Trial {self.trial_number}")
        self.trial_number -= 1
        self.trial_type = self.trials[self.trial_number]
        active_zones = next(target_generator)
        trial_thread = threading.Thread(target=self.run_trial_by_type, args=(self.trial_type, active_zones))
        trial_thread.daemon = True
        trial_thread.start()

    def stop_trial(self):
        if self.trial_number == 0:
            self.log("Warning: No trial has started yet. Exiting.")
            exit()
        # For simplicity, if stop is requested, we immediately stop the session.
        self.log("Stopping session...")
        self.cancelled = True
        self.sampling_active = False
        self.save_data('TCE_data_stopped')
        exit()

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
            self.log("Waiting 20 seconds between trials...")
            time.sleep(20)
            self.sampling_active = False
            self.trial_in_progress = False
            self.end_trial()

    def run_trial_hold_T1(self, active_zones):
        ramp_duration = (self.T1 - self.baseline) / self.ramp_speed
        self.thermode.set_durations([ramp_duration if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T1 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Ramping up to T1 ({self.T1}) over {ramp_duration:.2f}s")
        time.sleep(ramp_duration)
        hold_duration = 30
        self.thermode.set_durations([hold_duration if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T1 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Holding at T1 ({self.T1}) for {hold_duration:.2f}s")
        time.sleep(hold_duration)
        ramp_down_duration = (self.T1 - self.baseline) / self.return_speed
        self.thermode.set_durations([ramp_down_duration if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.baseline for _ in active_zones])
        self.thermode.stimulate()
        self.log(f"Ramping down to baseline ({self.baseline}) over {ramp_down_duration:.2f}s")
        time.sleep(ramp_down_duration)

    def run_trial_step_up(self, active_zones):
        ramp_duration_A = (self.T1 - self.baseline) / self.ramp_speed
        self.thermode.set_durations([ramp_duration_A if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T1 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Ramping up to T1 ({self.T1}) over {ramp_duration_A:.2f}s")
        time.sleep(ramp_duration_A)
        hold_duration_A = 5
        self.thermode.set_durations([hold_duration_A if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T1 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Holding at T1 ({self.T1}) for {hold_duration_A:.2f}s")
        time.sleep(hold_duration_A)
        ramp_duration_step = 1 / self.ramp_speed
        self.thermode.set_durations([ramp_duration_step if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T2 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Ramping up to T2 ({self.T2}) over {ramp_duration_step:.2f}s")
        time.sleep(ramp_duration_step)
        hold_duration_B = 5
        self.thermode.set_durations([hold_duration_B if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T2 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Holding at T2 ({self.T2}) for {hold_duration_B:.2f}s")
        time.sleep(hold_duration_B)
        self.thermode.set_durations([ramp_duration_step if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T1 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Ramping down to T1 ({self.T1}) over {ramp_duration_step:.2f}s")
        time.sleep(ramp_duration_step)
        hold_duration_C = 20
        self.thermode.set_durations([hold_duration_C if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T1 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Holding at T1 ({self.T1}) for {hold_duration_C:.2f}s")
        time.sleep(hold_duration_C)
        ramp_down_duration = (self.T1 - self.baseline) / self.return_speed
        self.thermode.set_durations([ramp_down_duration if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.baseline for _ in active_zones])
        self.thermode.stimulate()
        self.log(f"Ramping down to baseline ({self.baseline}) over {ramp_down_duration:.2f}s")
        time.sleep(ramp_down_duration)

    def run_trial_step_down(self, active_zones):
        ramp_duration_A = (self.T2 - self.baseline) / self.ramp_speed
        self.thermode.set_durations([ramp_duration_A if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T2 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Ramping up to T2 ({self.T2}) over {ramp_duration_A:.2f}s")
        time.sleep(ramp_duration_A)
        hold_duration_A = 5
        self.thermode.set_durations([hold_duration_A if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T2 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Holding at T2 ({self.T2}) for {hold_duration_A:.2f}s")
        time.sleep(hold_duration_A)
        ramp_duration_step = 1 / self.ramp_speed
        self.thermode.set_durations([ramp_duration_step if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T1 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Ramping down to T1 ({self.T1}) over {ramp_duration_step:.2f}s")
        time.sleep(ramp_duration_step)
        hold_duration_B = 5
        self.thermode.set_durations([hold_duration_B if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T1 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Holding at T1 ({self.T1}) for {hold_duration_B:.2f}s")
        time.sleep(hold_duration_B)
        self.thermode.set_durations([ramp_duration_step if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T2 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Ramping up to T2 ({self.T2}) over {ramp_duration_step:.2f}s")
        time.sleep(ramp_duration_step)
        hold_duration_C = 20
        self.thermode.set_durations([hold_duration_C if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T2 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Holding at T2 ({self.T2}) for {hold_duration_C:.2f}s")
        time.sleep(hold_duration_C)
        ramp_down_duration = (self.T2 - self.baseline) / self.return_speed
        self.thermode.set_durations([ramp_down_duration if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.baseline for _ in active_zones])
        self.thermode.stimulate()
        self.log(f"Ramping down to baseline ({self.baseline}) over {ramp_down_duration:.2f}s")
        time.sleep(ramp_down_duration)

    def run_trial_hold_T2(self, active_zones):
        ramp_duration = (self.T2 - self.baseline) / self.ramp_speed
        self.thermode.set_durations([ramp_duration if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T2 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Ramping up to T2 ({self.T2}) over {ramp_duration:.2f}s")
        time.sleep(ramp_duration)
        hold_duration = 30
        self.thermode.set_durations([hold_duration if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.T2 if active else self.baseline for active in active_zones])
        self.thermode.stimulate()
        self.log(f"Holding at T2 ({self.T2}) for {hold_duration:.2f}s")
        time.sleep(hold_duration)
        ramp_down_duration = (self.T2 - self.baseline) / self.return_speed
        self.thermode.set_durations([ramp_down_duration if active else 0 for active in active_zones])
        self.thermode.set_temperatures([self.baseline for _ in active_zones])
        self.thermode.stimulate()
        self.log(f"Ramping down to baseline ({self.baseline}) over {ramp_down_duration:.2f}s")
        time.sleep(ramp_down_duration)

    def end_trial(self):
        # Save trial samples and notes to file
        data_dir = os.path.join(os.path.dirname(__file__), "task_data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        file_name = f"{self.patient_info['name']}_{self.patient_info['date']}_temp_taskdata.csv"
        full_path = os.path.join(data_dir, file_name)
        file_exists = os.path.exists(full_path)
        try:
            with open(full_path, 'a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(["trial_count", "trial_type", "timestamp", "pain_rating", "temperature"])
                for sample in self.trial_samples:
                    writer.writerow(sample)
                writer.writerow([])  # blank line between sections
                writer.writerow(["timestamp", "note"])
                for note in self.notes:
                    writer.writerow(note)
            self.log(f"Trial {self.trial_number} data saved to {file_name}")
            self.trial_samples.clear()
            self.notes.clear()
        except Exception as e:
            self.log(f"Error saving trial data: {e}")

    def finish_session(self):
        self.sampling_active = False
        self.log("Finishing session and saving data...")
        self.save_data('TCE_data')
        self.log("Session complete.")

    def save_data(self, filename_suffix):
        data_dir = os.path.join(os.path.dirname(__file__), "task_data")
        base_name = f"{self.patient_info['name']}_{self.patient_info['date']}_{filename_suffix}"
        full_path = os.path.join(data_dir, base_name)
        filename = f"{full_path}.csv"
        n = 2
        while os.path.exists(filename):
            overwrite = input(f"File {filename} already exists. Overwrite? (y/n): ").lower()
            if overwrite != 'y':
                filename = f"{full_path}_{n}.csv"
                n += 1
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
                writer.writerow([])
                writer.writerow(['Experimenter Log'])
                # In CLI we assume print output serves as our log
                writer.writerow([f"Session completed at {self.get_current_time(time.time())}"])
            self.log(f"Session data saved to {filename}")
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except Exception as e:
            self.log(f"Could not save data: {e}")

def get_patient_calibration():
    # For CLI, get patient info via command-line inputs.
    name = input("Enter Patient Name: ").strip()
    if not name:
        print("Invalid name. Exiting.")
        exit()
    trial_start = input("Enter starting trial number (default 1): ").strip()
    try:
        trial_start = int(trial_start) if trial_start else 1
    except ValueError:
        print("Trial start must be a number. Exiting.")
        exit()
    date_str = time.strftime("%Y%m%d")
    # For simplicity, load calibration from a YAML file similar to the GUI version.
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    config_file = os.path.join(config_dir, f"{name}_{date_str}_config.yaml")
    try:
        import yaml
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading config file: {e}")
        exit()
    side = config.get("side", None)
    T1 = config.get("T1", None)
    if T1 is None:
        print("T1 value is missing in the config file. Exiting.")
        exit()
    T2 = T1 + 1
    com_port = config.get("COM_Port", None)
    if com_port is None:
        print("COM Port is missing in the config file. Exiting.")
        exit()
    num_blocks = config.get("num_blocks", None)
    patient_info = {"name": name, "date": date_str}
    return T1, T2, com_port, num_blocks, side, trial_start, patient_info

def update_config(config_file, num_blocks):
    trial_order_list = generate_trial_order(num_blocks)
    trial_order_mapping = {i+1: trial for i, trial in enumerate(trial_order_list)}
    try:
        import yaml
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        config['trial_order'] = trial_order_mapping
        with open(config_file, 'w') as file:
            yaml.dump(config, file, indent=4)
    except Exception as e:
        print(f"Error updating config: {e}")
    return trial_order_mapping

def main():
    T1, T2, com_port, num_blocks, side, trial_start, patient_info = get_patient_calibration()
    thermode = tcs.TcsDevice(port=com_port)
    thermode.set_quiet()
    thermode.set_baseline(baselineTemp)
    thermode.set_ramp_speed([ramp_speed] * 5)
    thermode.set_return_speed([return_speed] * 5)
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    config_file = os.path.join(config_dir, f"{patient_info['name']}_{patient_info['date']}_config.yaml")
    trial_order_mapping = update_config(config_file, num_blocks)
    experimenter = ExperimenterCLI(thermode, T1, T2, patient_info, side, trial_start,
                                   trial_order_mapping, baselineTemp, ramp_speed, return_speed)
    while True:
        command = input("\nEnter command: (N)ext trial, (R)epeat last trial, (A)dd note, or (S)top: ").strip().lower()
        if command == 'n':
            experimenter.start_trial()
        elif command == 'r':
            experimenter.repeat_last_trial()
        elif command == 'a':
            experimenter.add_note()
        elif command == 's':
            experimenter.stop_trial()
        else:
            print("Invalid command.")

if __name__ == "__main__":
    main()