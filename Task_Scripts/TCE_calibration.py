# Script for running the calibration before TCE task
"""
This script calibrates the appropriate temperatures for the TCE task and creates a config file 
that is later referenced in TCE_task.py
Author: Lucille Johnston
Updated: 05/16/25

Ramp and Hold Calibration 
- ramp and hold different temps to find temp associated with 50/100 pain rating
- automatically adjust temp of next trial based on rating during last 5 seconds of hold phase (30s total)

Creates temporary trial_data.csv file with timestamps, pain ratings, and temperatures
- this file is saved to calibration_data folder
- this file is deleted after the config and calibration_data.csv files are created

Creates config.yaml file with patient info, COM port, and calibrated T1 temp
Creates calibration_data.csv file with timestamps, pain ratings, and temperatures

"""

# Import necessary libraries
import time, csv, yaml, os, threading
import TcsControl_py3 as tcs 
import eVAS_slider as evas # eVAS slider
import tkinter as tk # for GUI
from tkinter import messagebox
from datetime import datetime
# from TTL_pulse import *
import subprocess


# Parameters
baselineTemp = 29 # baseline temperature in °C
ramp_speed = [1.5,1.5,1.5,1.5,1.5] # °C/s
return_speed = [6,6,6,6,6] # °C/s
#target_generator = tcs.target_zone_generator()  # Create a target zone generator
#thermode_port = 'COM7' ########## CHANGE TO MATCH DEVICE ##########
thermode_port = '/dev/tty.usbmodem11401'
#vas_port = 'COM6' ########## CHANGE TO MATCH DEVICE #########
vas_port = '/dev/tty.usbmodem11301'
# ttl_port_name = '/dev/tty.usbserial-BBTKUSBTTL'
# ttl_serial = open_serial_port(ttl_port_name)

"""
On windows: Task bar > USB hardware + media > open devices + printers > MCP2221
- Double click MCP2221 to open properties > hardware tab --> should have MCP2221 (COM ##) 
example: thermode_port = COM3

On Mac: Terminal > ls /dev/tty.*
- Should have something like /dev/tty.usbmodem11101
example: thermode_port = /dev/tty.usbmodem1401
"""

# # ########### FOR TROUBLESHOOTING ONLY ##########
# import virtual_QST as tcs
# thermode = tcs.VirtualThermode()
# com_port = 'virtual'
# # ###############################################

#Settings
thermode = tcs.TcsDevice(port=thermode_port)
thermode.set_quiet() # set device to quiet mode - shouldn't heat up between things
thermode.set_baseline(baselineTemp)
thermode.set_ramp_speed(ramp_speed)
thermode.set_return_speed(return_speed)
vas_slider = evas.eVASSlider(port=vas_port)

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

# --- PHASE 2 Calibration ---       
class ExperimenterGUI:
    def __init__(self, root, thermode, vas_slider, patient_info, baseline, ramp_speed, return_speed, hold_time, side):#, ttl_log):
        self.thermode = thermode
        self.patient_info = patient_info
        self.side = side
        self.vas_slider = vas_slider 
        # self.ttl_log = ttl_log
        # List to store continuous calibration samples (timestamp, pain, temp) 
        self.calibration_samples = []
        self.hold_phase_samples = [] #Buffer to store last 5 seconds of hold phase
        self.notes = []
        self.sampling_active = False
        self.sampling_interval = 10 # 10Hz = every 100ms //  100Hz = every 10ms

        # Temp Parameters
        self.baseline = baseline
        self.ramp_speed = ramp_speed
        self.return_speed = return_speed
        self.hold_time = hold_time
        self.target_temp = 38 # Start with 39 C
        self.current_trial_target = None
        self.cancelled = False
        self.trial_in_progress = False
        self.trial_completed = False  # Indicates that a trial has finished and is awaiting a rating.
        self.calibration_data = []  # List of dictionaries, one per trial.
        self.trial_count = 0
        self.target_temps = []
        self.calibrated_T1 = None
        self.trial_start_time = None
        self.maximum_temp_reached = False # Flag to indicate if max temp has been reached
        # Target zone generator
        #self.target_gen = target_generator
        
        # Build the GUI
        self.root = root
        self.root.geometry("1000x1000")
        self.root.title("Calibration: Experimenter Interface")
        
        # Log trial information 
        self.log_text = tk.Text(self.root, height=20, width=60, font=('Arial', 16))
        self.log_text.pack(pady=10)
        
        # Trial counter 
        self.trial_label = tk.Label(self.root, text="Trial: 0", font=('Arial', 16))
        self.trial_label.pack(pady=10)
        
        # Buttons frame
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        self.next_trial_button = tk.Button(btn_frame, text="Next Trial", font=('Arial',16), command=self.next_trial)
        self.next_trial_button.grid(row=0, column=0, padx=10, pady=10)

        self.repeat_trial_button = tk.Button(btn_frame, text="Repeat Last Trial", font=('Arial',16), command=self.repeat_last_trial)
        self.repeat_trial_button.grid(row=0, column=1, padx=10, pady=10)

        self.finish_button = tk.Button(btn_frame, text="Finish Calibration", font=('Arial', 16), command=self.finish_calibration)
        self.finish_button.grid(row=0, column=3, padx=10, pady=10)
        
        self.stop_button = tk.Button(btn_frame, text="Stop", font = ('Arial',16), command=self.stop_calibration)
        self.stop_button.grid(row=0, column=4, padx=10, pady=10)

        self.notes_label = tk.Label(self.root, text="Notes:", font=('Arial', 16))
        self.notes_label.pack(pady=10)
        self.notes_entry = tk.Entry(self.root, font=('Arial', 16))
        self.notes_entry.pack(pady=10)
        self.notes_entry.bind("<Return>", self.add_note)

        # # Send TTL to mark start of session
        # try:
        #     send_session_start_ttl(ttl_serial, self.ttl_log)
        # except Exception as e:
        #     self.log(f"Error sending TTL pulse: {e}")
    
    def log(self, message): # Append a message to the log text widget
        def append():
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
        self.root.after(0, append)
    
    def add_note(self, event):
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
        pain = self.vas_slider.get_value()
        temp = self.thermode.get_temperatures()
        if temp and isinstance(temp, list) and all(isinstance(t, float) for t in temp):
            self.calibration_samples.append((self.trial_count,current_time_str, pain, temp))
            # Buffer last 5 seconds of hold phase
            if getattr(self, 'in_hold_phase', False):
                self.hold_phase_samples.append((current_time, pain, temp))
                self.hold_phase_samples = [(time, pain, temp) for time, pain, temp in self.hold_phase_samples if current_time - time <= 5]
        if self.sampling_active:
            self.root.after(self.sampling_interval, self.sample_loop)

    def update_trial_label(self):
        def update():
            self.trial_label.config(text=f"Trial: {self.trial_count}")
        self.root.after(0, update)
    
    def next_trial(self):
        """
        If a trial is not in progress, start the next trial.
        """
        if self.trial_in_progress:
            return

        self.trial_start_time = datetime.now().strftime("%H:%M:%S")
        # Disable Next Trial button while the trial runs.
        self.next_trial_button.config(state=tk.DISABLED)
        self.repeat_trial_button.config(state=tk.DISABLED)
        self.finish_button.config(state=tk.DISABLED)
        trial_thread = threading.Thread(target=self.run_trial)
        trial_thread.daemon = True
        trial_thread.start()

    def repeat_last_trial(self):
        if self.trial_count == 0:
            messagebox.showerror("Error", "No trial to repeat")
            return
        self.log(f"Repeating Trial {self.trial_count} at {self.target_temp}C")
        self.target_temp -= 1
        # Disable all the buttons
        self.next_trial_button.config(state=tk.DISABLED)
        self.repeat_trial_button.config(state=tk.DISABLED)
        self.finish_button.config(state=tk.DISABLED)
        # Return to the same trial
        trial_thread = threading.Thread(target=self.run_trial)
        trial_thread.daemon = True
        trial_thread.start()

    def run_trial(self):
        """
        Runs one calibration trial: ramp-up, hold, ramp-down.
        Then enables the Record Rating button.
        """
        if self.cancelled:
            return
        self.trial_in_progress = True
        self.trial_count += 1
        self.update_trial_label()
        self.start_sampling()
        # Increment target temperature by 1°C.
        self.target_temp += 1
        if self.target_temp > 51:
            self.log("Maximum temperature reached. Finishing calibration.")
            self.maximum_temp_reached = True
            self.trial_in_progress = False
            self.finish_button.config(state=tk.NORMAL)
            return
        
        # try:
        #     send_trial_ttl(ttl_serial, self.ttl_log)
        # except Exception as e:
        #     self.log(f"Error sending TTL pulse: {e}")

        self.log(f"Trial {self.trial_count}: Target temperature = {self.target_temp}°C")
        self.ramp_durations = [(self.target_temp - self.baseline) / s for s in self.ramp_speed]
        self.ramp_down_durations = [(self.target_temp - self.baseline) / d for d in self.return_speed]
        self.active_zones = [1,1,1,1,1] #next(self.target_gen)
        self.log("Active zone: " + str(self.active_zones))
        self.ramp_up_and_hold_phase()
        
    def ramp_up_and_hold_phase(self): # --- Ramp-Up and Hold Phase ---
        self.log(f"Ramp-up + Hold: Stimulating {self.target_temp}°C for {self.ramp_durations[0]:.2f}+{self.hold_time}s")
        ramp_hold_duration = self.ramp_durations[0] + self.hold_time
        # Send temp to thermode
        self.thermode.set_durations([ramp_hold_duration if active else 0 for active in enumerate(self.active_zones)])
        self.thermode.set_temperatures([self.target_temp if active else self.baseline for active in self.active_zones])
        self.thermode.stimulate()
        # Schedule TTL event at the beginning of the hold phase
        hold_phase_start_delay = int(self.ramp_durations[0] * 1000)  # end of ramp-up, start of hold
        # self.root.after(hold_phase_start_delay, lambda: send_event_ttl(ttl_serial, self.ttl_log))
        
        # make in_hold_phase True for last 5 seconds of hold phase (after ramp duration + 25s)
        self.root.after(int((self.ramp_durations[0] + 25) * 1000), lambda: setattr(self, "in_hold_phase", True))

        # Schedule TTL event at the end of the hold phase
        hold_phase_end_delay = int((self.ramp_durations[0] + self.hold_time) * 1000)
        # self.root.after(hold_phase_end_delay, lambda: send_event_ttl(ttl_serial, self.ttl_log))
    
        total_duration = ramp_hold_duration + self.ramp_down_durations[0]
        delay_ms = int(total_duration * 1000)
        self.root.after(delay_ms, self.finalize_trial)
    
    def finalize_trial(self):
        if self.cancelled:
            self.trial_in_progress = False
            return
        
        # try: # send a TTL pulse to mark end of the trial
        #     send_trial_ttl(ttl_serial, self.ttl_log)
        # except Exception as e:
        #     self.log(f"Error sending TTL pulse: {e}")

        # Save the target temperature for this trial.
        self.current_trial_target = self.target_temp
        # Mark that the trial has completed and is awaiting rating.
        self.trial_completed = True
        # The Next Trial button remains disabled until a rating is recorded.
        self.trial_in_progress = False
        self.sampling_active = False
        self.compute_rating()
        self.log("Ending continous sampling.")
        self.flush_trial_data()
        self.sampling_active = False # Stop sampling
        self.log("Waiting 10s before next trial.")
        self.root.after(10000, self.enable_buttons) # Wait 10s before enabling buttons

    def flush_trial_data(self): # save trial data to file
        calibration_data_dir = os.path.join(os.path.dirname(__file__), "calibration_data")
        file_path = os.path.join(calibration_data_dir, f"{self.patient_info['name']}_{self.patient_info['date']}_temp_trialdata.csv")
        file_exists = os.path.exists(file_path)
        try:
            with open(file_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(['trial_count','timestamp', 'pain_rating', 'temperature'])
                for sample in self.calibration_samples:
                    writer.writerow(sample)
                for note in self.notes:
                    writer.writerow(note)
            self.log(f"Trial {self.trial_count} data saved to temp_trialdata")
            self.calibration_samples.clear()
            self.hold_phase_samples.clear()
            self.notes.clear()
        except Exception as e:
            self.log(f"Error saving trial data: {e}")

    def compute_rating(self):
        """
        Compute average pain rating over the last 5 seconds of hold phase 
        """
        if self.hold_phase_samples:
            avg_rating = sum(sample[1] for sample in self.hold_phase_samples) / len(self.hold_phase_samples)
            self.adjust_temperature(avg_rating)
    
    def adjust_temperature(self, rating):
        """
        Adjust target temp based on pain rating
        """
        if rating < 40:
            self.log(f"Pain rating {rating:.2f}. Increasing temp to {self.target_temp+1}")
        elif rating >= 60:
            self.log(f"Pain rating {rating:.2f}. Decreasing temp to {self.target_temp-1}")
            self.target_temp -=2 # Decrease temp by 1C
        elif 40 <= rating <= 60:
            self.log(f"Pain rating {rating:.2f} within range (40-60).\n")
            self.log("Either Finish Calibration, Repeat Trial, or press Next Trial to increase temp again")
            self.target_temps.append((self.current_trial_target, rating))
        self.trial_completed = False # Mark that the trial has been recorded
    
    def enable_buttons(self):
        # Update all of the buttons in the GUI
        self.next_trial_button.config(state=tk.NORMAL) # Enable Next Trial button
        self.repeat_trial_button.config(state=tk.NORMAL) # Enable Repeat Trial Button
        self.finish_button.config(state=tk.NORMAL) # Enable Finish Calibration Button
        self.log("Ready to start next trial.")

    def finish_calibration(self):
        self.sampling_active = False
        self.log("Finishing sampling calibration data.")

        # try:
        #     send_session_end_ttl(ttl_serial, self.ttl_log)
        # except Exception as e:
        #     self.log(f"Error sending TTL pulse: {e}")

        self.save_data('calibration_data')
        self.save_config()
    
    def stop_calibration(self):
        self.sampling_active = False
        self.log("Calibration stopped by experimenter")

        # try:
        #     send_session_end_ttl(ttl_serial, self.ttl_log)
        # except Exception as e:
        #     self.log(f"Error sending TTL pulse: {e}")

        self.save_data("calibration_data_stopped")
        self.root.destroy()
    
    def save_data(self, filename_suffix): 
        calibration_data_dir = os.path.join(os.path.dirname(__file__), "calibration_data")
        file = f"{self.patient_info['name']}_{self.patient_info['date']}_{filename_suffix}"
        full_path = os.path.join(calibration_data_dir,file)
        filename = f"{full_path}.csv"
        n=2
        while os.path.exists(filename):
            if not messagebox.askyesno("Overwrite File", f"File {file} already exists. Overwrite?"):
                filename = f"{full_path}_{n}.csv"
                n+=1
            else:
                break
        try:
            temp_file_path = os.path.join(calibration_data_dir, f"{self.patient_info['name']}_{self.patient_info['date']}_temp_trialdata.csv")
            if os.path.exists(temp_file_path):
                with open(temp_file_path, 'r') as src:
                    data = src.read()
                with open(filename, 'w', newline='') as dst:
                    dst.write(data)
            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([]) # add blank line to differentiate
                writer.writerow(['Experimenter Log'])
                log_content = self.log_text.get("1.0", tk.END).splitlines()
                for line in log_content:
                    writer.writerow([line])
                        
            self.log(f"Saved calibration data to {file}")
            messagebox.showinfo("Data Saved", f"Calibration data saved as: {file}")
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except Exception as e:
            messagebox.showerror("File Error", f"Could not save calibration data: {e}")
        
    def save_config(self):
        if self.trial_in_progress or self.trial_completed:
            messagebox.showwarning("Trial Pending", "Please complete the current trial and record its rating before finishing calibration.")
            return
        temp_ratings = {}
        
        if self.maximum_temp_reached:
            self.calibrated_T1 = 49 # If max temp reached, set T1 to 49C
            self.log(f"Selected T1: {self.calibrated_T1} C (Max temp reached)")
        else:
            for temp, rating in self.target_temps:
                if temp not in temp_ratings:
                    temp_ratings[temp] = []
                temp_ratings[temp].append(rating)
            avg_temp_ratings = {temp: sum(ratings)/len(ratings) for temp, ratings in temp_ratings.items()}
            self.calibrated_T1 = min(avg_temp_ratings.keys(), key=lambda temp: abs(avg_temp_ratings[temp] - 50))
            self.log(f"Selected T1: {self.calibrated_T1} C (Avg rating: {avg_temp_ratings[self.calibrated_T1]:.2f})")
        
        # Create trial order
        num_blocks = 10 
        config = {
            "patient_info": self.patient_info,
            "Thermode_Port": thermode_port,
            "VAS_Port": vas_port,
            "T1": self.calibrated_T1,
            "side": self.side,
            "calibration_data_file": f"{self.patient_info['name']}_{self.patient_info['date']}_calibration_data.csv",
            "num_blocks": num_blocks, # Number of blocks of trials 
        }
        config_dir = os.path.join(os.path.dirname(__file__), "configs")
        file_name = f"{self.patient_info['name']}_{self.patient_info['date']}_config.yaml"
        full_path = os.path.join(config_dir,file_name)
        if os.path.exists(full_path):
            if not messagebox.askyesno("Overwrite File", f"File {file_name} already exists. Overwrite?"):
                return
        try:
            with open(full_path, "w") as f:
                yaml.dump(config, f, indent=4)
            messagebox.showinfo("Calibration Complete", f"Calibration saved as: {file_name}")
        except Exception as e:
            messagebox.showerror("File Error", f"Could not save configuration: {e}")
        self.root.destroy()
    
    def run(self):
        self.root.mainloop()

# --- Main ---
if __name__ == '__main__':
    # First, temporary Tk root for patient info GUI
    info_root = tk.Tk()
    patient_info_gui = PatientInfoGUI(info_root)
    info_root.mainloop()

    # Retrieve patient info 
    patient_info = patient_info_gui.patient_info
    side = patient_info_gui.side
    if not patient_info:
        exit("No patient info provided.")
    # # Ramp + Hold Calibration 
    # ttl_filename = f"{patient_info['name']}_{patient_info['date']}_ttl_log.csv"
    # ttl_log_path = os.path.join(os.path.dirname(__file__), "calibration_data", ttl_filename)
    # ttl_log = open(ttl_log_path, 'a')
    root = tk.Tk()
    experimenter_gui = ExperimenterGUI(root, thermode, vas_slider, patient_info,
                                       baseline=baselineTemp, ramp_speed = ramp_speed,
                                       return_speed = return_speed, hold_time = 30, side=side) #, ttl_log=ttl_log)
    experimenter_gui.run()


