"""
Have the patient move the slider around to get used to it.
Shows a GUI so the experimenter can see the slider value in real time.
Sampling runs until the user clicks "Stop Sampling."
Data is then saved as a .csv file.
"""

import time, csv, os, yaml, sys
from datetime import datetime
import tkinter as tk
import eVAS_slider as evas

class SliderMovementGUI:
    def __init__(self, root, config):
        self.root = root
        self.config = config
        self.patient = config['patient_info']['name']
        self.date = config['patient_info']['date']
        self.vas_port = config['VAS_port']
        self.csv_filename = f"{self.patient}_{self.date}_slider_data.csv"
        self.vas_slider = evas.eVASSlider(port=self.vas_port)
        self.data = []
        self.sampling_interval = 10  # ms (approx 100Hz)
        self.sampling_active = True
        self.create_widgets()

    def create_widgets(self):
        self.root.title("Slider Movement Sampling")
        self.info_label = tk.Label(self.root, text="Move the slider. Data is sampling...", font=("Arial",16))
        self.info_label.pack(pady=20)
        self.stop_button = tk.Button(self.root, text="Stop Sampling", font=("Arial",16), command=self.stop_sampling)
        self.stop_button.pack(pady=20)

    def sample_slider(self):
        if self.sampling_active:
            value = self.vas_slider.get_value()
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.data.append([timestamp, value])
            self.root.after(self.sampling_interval, self.sample_slider)
        else:
            self.save_data()
            self.root.destroy()
    def stop_sampling(self):
        self.sampling_active = False

    def save_data(self):
        output_dir = os.path.join(os.path.dirname(__file__), "task_data")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir, self.csv_filename)
        try:
            with open(file_path, "w", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(["timestamp", "slider_value"])
                csv_writer.writerows(self.data)
            print(f"Slider data saved to {file_path}")
        except Exception as e:
            print(f"Error saving slider data: {e}")

def main(config_file):
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)
    root = tk.Tk()
    gui = SliderMovementGUI(root, config)
    root.after(gui.sampling_interval, gui.sample_slider)
    root.mainloop()

if __name__ == "__main__":
    import os
    import tempfile
    from datetime import datetime
    import yaml

    if len(sys.argv) < 2:
        patient = input("Enter patient name: ")
        today = datetime.today().strftime("%Y-%m-%d")
        save_path = input("Enter directory to save CSV (default: 'task_data' directory beside this file): ").strip()
        if not save_path:
            save_path = os.path.join(os.path.dirname(__file__), "task_data")
        # Create a configuration dictionary
        config = {
            'patient_info': {
                'name': patient,
                'date': today
            },
            'VAS_port': '/dev/tty.usbmodem11301',  # update this as needed
            'csv_save_path': save_path
        }
        # Write the configuration to a temporary YAML file
        config_file = os.path.join(tempfile.gettempdir(), "slider_config.yaml")
        with open(config_file, "w") as f:
            yaml.dump(config, f)
        print(f"Generated temporary config file: {config_file}")
    else:
        config_file = sys.argv[1]
    main(config_file)