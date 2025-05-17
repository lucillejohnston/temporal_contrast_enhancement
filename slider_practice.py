# Import necessary libraries
import time, math
import tkinter as tk # for GUI
import threading
import VAS_slider as vas

# --- PHASE 1 Practice Slider ---
class SliderPracticeGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Slider Practice")
        self.root.geometry("800x600")
        self.label_instruction = tk.Label(self.root, 
                                          text="Please follow the movement with the slider.", 
                                          font=('Arial', 20))
        self.label_instruction.pack(pady=10)

        # Error buffering variables
        self.error_buffer = []
        self.error_window = 10 # seconds over which to average
        self.error_threshold = 3.5 # threshold to enable proceed button 

        # For saving samples (timestamp, control_slider, instruction_slider, error)
        self.samples = []
        self.sampling_interval = 100 # sampling interval in ms (10Hz=100ms)
        self.continue_sampling = True # flag to continue sampling

        # start VASSlider sampling in dedicated thread
        self.start_slider_sampling()
        self.start_time = time.time()

        sliders_frame = tk.Frame(self.root)
        sliders_frame.pack(pady=20)

        # Instruction Slider to read temp in real time 
        # Read only, updated periodically 
        self.instruction_slider = tk.Scale(sliders_frame, from_=100, to=0, orient='vertical', 
                                       length=400, font=('Arial', 20), state='disabled')
        self.instruction_slider.pack(side='left', padx=10)
        self.instruction_slider.set(0)

        # Control Slider to show the current value
        self.control_slider = tk.Scale(sliders_frame, from_=100, to=0, orient='vertical',
                                        length=400, font=('Arial', 20), state='disabled')
        self.control_slider.pack(side='left', padx=30)
        self.control_slider.set(0)

        self.proceed_button = tk.Button(self.root, text="Proceed", 
                                        font=('Arial', 20), command=self.proceed, state='disabled')
        self.proceed_button.pack(pady=20)
        # Start periodic sampling/updating 
        self.periodic_update()
    
    def start_slider_sampling(self):
        self.vas_slider = vas.VASSlider()
        self.slider_thread = threading.Thread(target=self.sample_slider_values, daemon=True)
        self.slider_thread.start()

    def sample_slider_values(self):
        while self.continue_sampling:
            result = self.vas_slider.get_value('both')
            if result is not None:
               value = result
               self.root.after(0, lambda v=value: self.update_control_slider(v))
            time.sleep(0.01)

    def update_control_slider(self, value):
         self.control_slider.config(state='normal')
         self.control_slider.set(value)
         self.control_slider.config(state='disabled')
   
    def update_instruction_slider(self):
        if not self.continue_sampling:
            return
        current_time = time.time()
        # Simulate automatic movemement using sine wave
        # Oscillate between 0 and 100 with varying speed
        auto_value = int(50 + 50 * math.sin(current_time))

        # Update display slider
        self.instruction_slider.config(state='normal')
        self.instruction_slider.set(auto_value)
        self.instruction_slider.config(state='disabled')
        
    def get_current_time(self, current_time):
        current_time_str = time.strftime("%H:%M:%S", time.localtime(current_time))
        milliseconds = int((current_time - int(current_time)) * 1000)
        return f"{current_time_str}.{milliseconds:03d}"

    def check_error(self):
        if time.time() - self.start_time < 2 or self.control_slider.get() == 0: # only start checking error after 2 seconds have passed and the slider has moved
            return
        # Calculate error between control slider and display slider
        instruction_val = self.instruction_slider.get()
        control_val = self.control_slider.get()
        error = abs(instruction_val - control_val)
        # Record with a timestamp 
        self.error_buffer.append((time.time(), error))
        self.error_buffer = [(t,e) for t, e in self.error_buffer if time.time() - t <= self.error_window]

        # Compute average error over the current window
        if self.error_buffer:
            mean_error = sum(e for t, e in self.error_buffer) / len(self.error_buffer) 
        else:
            mean_error = float('inf')
        
        # Enable proceed button if average error is below threshold
        if mean_error < self.error_threshold:
            self.proceed_button.config(state='normal')
            self.label_instruction.config(text="Great job! Proceed to the next step.")
            self.continue_sampling = False # stop further updates
        else:
            self.proceed_button.config(state='disabled')

    def periodic_update(self):
        if not self.continue_sampling:
            return
        self.update_instruction_slider()
        self.check_error()
        self.root.after(self.sampling_interval, self.periodic_update)

    def proceed(self):
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# --- Main Program ---
if __name__ == '__main__':
    sliderpractice_gui = SliderPracticeGUI()
    sliderpractice_gui.run()