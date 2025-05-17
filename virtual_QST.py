import time
import sys

"""
Made by Lucy to troubleshoot any tasks without the physical QST.Labs device
virtual_QST.py

This module simulates a thermode for testing BehaviorTaskMaster code without an actual device.
It offers an interface almost identical to the real thermode control, but instead prints
and computes a simulated temperature, which ramps from an ambient value to a target temperature.
"""

class VirtualThermode:
    def __init__(self, ambient_temp=25.0, ramp_duration=10.0):
        self.ambient_temp = ambient_temp
        self.ramp_duration = ramp_duration
        self.target_temp = None
        self.start_time = None
        self.messages = [] 

    def log_message(self, message):
        self.messages.append(message)

    def connect(self):
        """Simulate connecting to the thermode."""
        self.log_message("Virtual thermode connected.")

    def set_quiet(self):
        self.log_message("Quiet mode set.")
    
    def set_baseline(self, baselineTemp):
        if baselineTemp > 40:
            baselineTemp = 40
        if baselineTemp < 20:
            baselineTemp = 20
        self.ambient_temp = baselineTemp
        self.log_message(f"Baseline temperature set to {self.ambient_temp:.2f}°C.")

    def set_durations(self, durations):
        for i in range(5):
            if durations[i] > 99.999:
                durations[i] = 99.999
            if durations[i] < 0.001:
                durations[i] = 0.001
        if durations.count(durations[0]) == len(durations):
            self.log_message(f"Stimulus duration set to {durations[0]:.3f}s for all zones.")
        else:
            for i in range(5):
                self.log_message(f"Stimulus duration for zone {i+1} set to {durations[i]:.3f}s.")

    def set_ramp_speed(self, rampSpeeds):
        for i in range(5):
            if rampSpeeds[i] > 300:
                rampSpeeds[i] = 300
            if rampSpeeds[i] < 0.1:
                rampSpeeds[i] = 0.1
        if rampSpeeds.count(rampSpeeds[0]) == len(rampSpeeds):
            self.log_message(f"Ramp speed set to {rampSpeeds[0]:.1f}°C/s for all zones.")
        else:
            for i in range(5):
                self.log_message(f"Ramp speed for zone {i+1} set to {rampSpeeds[i]:.1f}°C/s.")
    
    def set_return_speed(self, returnSpeeds):
        for i in range(5):
            if returnSpeeds[i] > 300:
                returnSpeeds[i] = 300
            if returnSpeeds[i] < 0.1:
                returnSpeeds[i] = 0.1
        if returnSpeeds.count(returnSpeeds[0]) == len(returnSpeeds):
            self.log_message(f"Return speed set to {returnSpeeds[0]:.1f}°C/s for all zones.")
        else:
            for i in range(5):
                self.log_message(f"Return speed for zone {i+1} set to {returnSpeeds[i]:.1f}°C/s.")

    def set_temperatures(self, temps):
        """
        Simulate setting a target temperature.
        Begins ramping from the ambient temperature to the target temperature.
        """
        for i in range(5):
            if temps[i] > 60:
                temps[i] = 60
            if temps[i] < 0.1:
                temps[i] = 0.1
        if temps.count(temps[0]) == len(temps):
            self.log_message(f"Target temperature set to {temps[0]:.2f}°C for all zones.")
            self.target_temps = [temps[0]] * 5
            self.start_time = time.time()
        else:
            for i in range(5):
                self.log_message(f"Target temperature for zone {i+1} set to {temps[i]:.2f}°C.")
            self.target_temps = temps
            self.start_time = time.time()
    
    def stimulate(self):
        # Print out the accumulated messages and then clear them.
        if self.messages:
            for m in self.messages:
                print(m)
        else:
            print("No messages accumulated.")
        # Then clear the messages.
        self.messages = []

    def get_temperatures(self):
        """
        Return the simulated current temperature.
        Temperature ramps over ramp_duration seconds from ambient_temp to target_temp.
        """
        if not hasattr(self, "target_temps") or self.start_time is None:
            print("No target temperature set. Returning ambient temperature.")
            return [self.ambient_temp] * 5

        elapsed = time.time() - self.start_time
        factor = min(elapsed / self.ramp_duration, 1.0)
        current_temps = [self.ambient_temp + factor * (target - self.ambient_temp)
                       for target in self.target_temps]
        print(f"Current temperatures: {[f'{temp:.2f}°C' for temp in current_temps]} (elapsed: {elapsed:.1f}s)")
        return current_temps

def main():
    vt = VirtualThermode()
    vt.connect()

    # Get target temperature from command-line argument if provided.
    if len(sys.argv) > 1:
        try:
            target = float(sys.argv[1])
        except ValueError:
            print("Invalid command-line temperature argument. Using default of 40°C.")
            target = 40.0
    else:
        print("No target temperature provided. Using default of 40°C.")
        target = 40.0

    vt.set_temperature(target)

    # Simulate periodic temperature sampling (e.g., every second for 15 seconds)
    duration = 15
    for i in range(duration):
        vt.get_current_temperature()
        time.sleep(1)

    print("Virtual thermode simulation complete.")

if __name__ == '__main__':
    main()

####################### LUCY MADE FUNCTIONS BELOW #############################
import random
def target_zone_generator():
    """
    Generates boolean list to indicate which zone is active
    
    - in each block of 5 trials, all 5 contacts are used exactly once
    - the same zone is never active in two consecutive trials 
    
    Returns: list of 5 boolean values
    """
    last_zone = None
    while True:
        block = list(range(5))
        random.shuffle(block)
        if last_zone is not None and block[0] == last_zone:
            # swap with the first zone in the block that is different
            for i in range(1, len(block)):
                if block[i] != last_zone:
                    block[0], block[i] = block[i], block[0]
                    break
        for zone in block:
            targets = [False] * 5
            targets[zone] = True
            last_zone = zone
            yield targets 