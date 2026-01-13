"""
Not currently working
"""

import time
import tkinter as tk
from tkinter import filedialog
import threading
import pandas as pd

class SliderGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Slider Practice")
        self.root.geometry("800x600")
        
        self.label_instruction = tk.Label(
            self.root, 
            text="Load a CSV and watch the slider move.",
            font=('Arial', 20)
        )
        self.label_instruction.pack(pady=10)
        
        # For storing CSV pain ratings and keeping track of index
        self.pain_values = []  # List of pain_rating values (0-100)
        self.current_index = 0
        
        self.sampling_interval = 100  # ms (10Hz sampling)
        self.continue_sampling = True  # flag to continue updating the slider
        
        # Frame for slider
        slider_frame = tk.Frame(self.root)
        slider_frame.pack(pady=20)
        
        # Single slider to display CSV values
        self.display_slider = tk.Scale(
            slider_frame, from_=100, to=0, orient='vertical',
            length=400, font=('Arial', 20), state='disabled'
        )
        self.display_slider.pack(padx=10)
        self.display_slider.set(0)
        
        # Button to load CSV file and begin slider movement
        self.load_button = tk.Button(self.root, text="Load CSV", font=('Arial', 16), command=self.load_csv)
        self.load_button.pack(pady=10)
        
        # Proceed button (enabled when finished playing CSV, if desired)
        self.proceed_button = tk.Button(
            self.root, text="Proceed", font=('Arial', 20), command=self.proceed, state='disabled'
        )
        self.proceed_button.pack(pady=20)
        
        # Start periodic update for slider movement
        self.periodic_update()
    
    def load_csv(self):
        filepath = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv")],
            initialdir="."  # Change if needed
        )
        if filepath:
            try:
                df = pd.read_csv(filepath)
                if "pain_rating" not in df.columns:
                    self.label_instruction.config(text="Error: 'pain_rating' column not found.")
                    return
                # Read and clip pain_rating values (assumed to be between 0 and 100)
                self.pain_values = df["pain_rating"].clip(lower=0, upper=100).tolist()
                self.current_index = 0
                self.label_instruction.config(text=f"Loaded {len(self.pain_values)} samples.")
                # Optionally enable the proceed button when finished playing all values.
                self.proceed_button.config(state='disabled')
            except Exception as e:
                self.label_instruction.config(text=f"Error loading CSV: {e}")
    
    def update_display_slider(self):
        # If CSV data are loaded and not finished, update slider with next value.
        if self.pain_values and self.current_index < len(self.pain_values):
            value = int(self.pain_values[self.current_index])
            self.display_slider.config(state='normal')
            self.display_slider.set(value)
            self.display_slider.config(state='disabled')
            self.current_index += 1
        # When finished, enable the proceed button.
        elif self.pain_values and self.current_index >= len(self.pain_values):
            self.label_instruction.config(text="All samples played.")
            self.proceed_button.config(state='normal')
            self.continue_sampling = False
    
    def periodic_update(self):
        if self.continue_sampling:
            self.update_display_slider()
            self.root.after(self.sampling_interval, self.periodic_update)
    
    def proceed(self):
        self.root.destroy()
    
    def run(self):
        self.root.mainloop()

# --- Main Program ---
if __name__ == '__main__':
    slider_gui = SliderGUI()
    slider_gui.run()