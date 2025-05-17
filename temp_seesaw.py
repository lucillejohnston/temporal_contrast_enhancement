"""
This script runs the seesaw task. 

The temperature ramps up and down between various target temperatures at the same rate
for all 5 contacts on the thermode.

Patients are asked to move the slider corresponding to the change in sensation that they feel
as the temperature moves (IMPORTANTLY NOT PAIN)

Used as part of the demo_day series of tasks

##### NOTE #####
Parameters I chose randomly and are subject to change:
ramp_speed
target_temps (BOTH HOT AND COLD???)
doing all 5 contacts at the same time
number of target_temps 
"""
# Import necessary libraries
import time, csv, os, sys, yaml
import VAS_slider as vas
import tkinter as tk # for GUI
from tkinter import messagebox
import TcsControl_py3 as tcs

if len(sys.argv) >=2:
    config_file = sys.argv[1]
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                com_port = config.get('COM_port', None)
                patient_info = config.get('patient_info', {})
                side = config.get('side', None)
        except Exception as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)
    else:
        print(f"Config file {config_file} not found.")
        sys.exit(1)

# Parameters
baselineTemp = 29 # baseline temperature in °C
ramp_speed = [3,3,3,3,3] # °C/s (ramp up + down at same rate) ###### PICKED RANDOMLY CAN ADJUST ######
target_generator = tcs.target_zone_generator()  # Create a target zone generator
"""
On windows: Task bar > USB hardware + media > open devices + printers > MCP2221
- Double click MCP2221 to open properties > hardware tab --> should have MCP2221 (COM ##) 
example: com_port = COM3

On Mac: Terminal > ls /dev/tty.*
- Should have something like /dev/tty.usbmodem11101
example: com_port = /dev/tty.usbmodem1401
"""

#Settings
thermode = tcs.TcsDevice(port=com_port) 
thermode.set_quiet() # set device to quiet mode - shouldn't heat up between things
thermode.set_baseline(baselineTemp)
thermode.set_ramp_speed(ramp_speed)

class ExperimenterGUI:
    def __init__(self, patient_info, root, thermode, side, baseline, ramp_speed):
        self.root = root
        self.thermode = thermode
        self.vas_slider = vas.VASSlider()
        self.patient_info = patient_info 
        self.side = side

        # List to store continuous seesaw samples (timestamp, pain, temp) 
        self.seesaw_samples = []
        self.notes = []
        self.sampling_active = False
        self.sampling_interval = 100 # 10Hz = every 100ms

        # Temp Parameters
        self.baseline = baseline
        self.ramp_speed = ramp_speed
        self.target_temps = [39, 25, 41, 23, 43, 21, 39, 23, 41, 25] # I did this very randomly 
        self.cancelled = False
        self.seesaw_data = []  # List of dictionaries
        self.seesaw_active = False
        # Target zone generator
        self.target_gen = target_generator
        
        # Build the GUI
        self.root.geometry("1000x1000")
        self.root.title("Seesaw: Experimenter Interface")
        
        # Log information 
        self.log_text = tk.Text(self.root, height=20, width=60, font=('Arial', 16))
        self.log_text.pack(pady=10)
        
        # Buttons frame
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        self.start_button = tk.Button(btn_frame, text='Start Task', font=('Arial', 16), command=self.start_seesaw)
        self.start_button.grid(row=0, column=0, padx=10, pady=10)

        self.finish_button = tk.Button(btn_frame, text="Finish Seesaw", font=('Arial', 16), command=self.finish_seesaw)
        self.finish_button.grid(row=0, column=1, padx=10, pady=10)
        
        self.stop_button = tk.Button(btn_frame, text="Stop", font = ('Arial',16), command=self.stop_seesaw)
        self.stop_button.grid(row=0, column=2, padx=10, pady=10)

        self.notes_label = tk.Label(self.root, text="Notes:", font=('Arial', 16))
        self.notes_label.pack(pady=10)
        self.notes_entry = tk.Entry(self.root, font=('Arial', 16))
        self.notes_entry.pack(pady=10)
        self.notes_entry.bind("<Return>", self.add_note)
    
    def log(self, message):
        # Append a message to the log text widget
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

    def start_seesaw(self):
        self.log("Starting seesaw task")
        # Begin continuous sampling (temperatures, ratings, etc.)
        self.start_sampling()
        self.seesaw_active = True
        self.thermode.set_temperatures([self.baseline] * 5)
        self.run_target_sequence(0)
    
    def run_target_sequence(self, index):
        if not self.seesaw_active or index >= len(self.target_temps):
            self.log("Completed seesaw sequence")
            return  
        target = self.target_temps[index]
        if index == 0:
            start_temp = self.baseline
        else:
            start_temp = self.target_temps[index-1]
        ramp_duration = abs(target - start_temp) / self.ramp_speed[0] if self.ramp_speed[0] != 0 else 0
        self.log(f"Ramp: ramping from {start_temp} to {target} in {ramp_duration:.1f} s")

        self.thermode.set_durations([ramp_duration] * 5)
        self.thermode.set_temperatures([target] * 5)
        self.thermode.stimulate()

        self.root.after(int(ramp_duration * 1000), lambda: self.run_target_sequence(index + 1))

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
        sensation = self.vas_slider.get_value(side)
        temp = self.thermode.get_temperatures()
        # If get_temperatures returns a single number, convert it to a 1-element list.
        if isinstance(temp, (float, int)):
            temp = [float(temp)]
        if isinstance(temp, list) and len(temp) == 5 and all(isinstance(t, float) for t in temp):
            avg_temp = sum(temp) / 5.0
            self.seesaw_samples.append((current_time_str, sensation, avg_temp))
        elif temp and isinstance(temp, list) and all(isinstance(t, float) for t in temp):
            self.seesaw_samples.append((current_time_str, sensation, temp))
        else:
            self.log(f"Invalid temperature data at {current_time_str}: {temp}")
        self.root.after(self.sampling_interval, self.sample_loop)

    def finish_seesaw(self):
        self.sampling_active = False
        self.log("Finishing sampling seesaw data")
        self.save_data('seesaw_data')
    
    def stop_seesaw(self):
        self.sampling_active = False
        self.log("Seesaw stopped by experimenter")
        self.save_data("seesaw_data_stopped")
        self.root.destroy()

    def save_data(self, filename_suffix): 
        save_dir = os.path.join(os.path.dirname(__file__), "demo_day_data")
            # Ensure the directory exists.
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        base_filename = os.path.join(save_dir, f"{self.patient_info['name']}_{self.patient_info['date']}_{filename_suffix}")
        filename = f"{base_filename}.csv"
        n=2
        while os.path.exists(filename):
            if not messagebox.askyesno("Overwrite File", f"File {filename} already exists. Overwrite?"):
                filename = f"{base_filename}_{n}.csv"
                n+=1
            else:
                break
        try:
            with open(filename, "w", newline="") as csvfile:
                # Write seesaw sample data 
                writer = csv.writer(csvfile)
                writer.writerow(['timestamp','sensation_rating','temperature'])
                for sample in self.seesaw_samples:
                    writer.writerow(sample)
                # Write notes (if any)
                writer.writerow([]) # add blank line to differentiate 
                writer.writerow(['timestamp','note'])
                for note in self.notes:
                    writer.writerow(note)
                
                # Write experimenter log 
                log_content = self.log_text.get("1.0", tk.END).splitlines()
                if log_content:
                    writer.writerow([]) # add blank line to differentiate
                    writer.writerow(['Experimenter Log'])
                    for line in log_content:
                        writer.writerow([line])
            self.log(f"Saved seesaw data to {filename}")
            messagebox.showinfo("Data Saved", f"Seesaw data saved as: {filename}")
        except Exception as e:
            messagebox.showerror("File Error", f"Could not save seesaw data: {e}")
        self.root.destroy()
    
    def run(self):
        self.root.mainloop()

# --- Main ---
if __name__ == '__main__':
    root = tk.Tk()
    experimenter_gui = ExperimenterGUI(patient_info, root, thermode, side, baselineTemp, ramp_speed)
    experimenter_gui.run()

