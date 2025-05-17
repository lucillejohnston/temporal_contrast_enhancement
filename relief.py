import TcsControl_py3 as tcs
import time

def main():
    try:
        inp = input("Enter desired temperature in °C (default 20): ").strip()
        target_temp = float(inp) if inp else 20.0
    except ValueError:
        print("Invalid input. Using default temperature of 20°C.")
        target_temp = 18.0

    # Initialize the TCS device on COM3 (adjust port to match your setup)
    thermode = tcs.TcsDevice(port='COM7')
    thermode.set_quiet()  # Set device to quiet mode to avoid unwanted heating between commands

    # Set baseline and ramp speed parameters
    baseline_temp = 29  # baseline temperature in °C
    ramp_speed = [2.0]*5  # Slow ramp speed (°C/s); adjust format as needed by your device
    thermode.set_baseline(baseline_temp)
    thermode.set_ramp_speed(ramp_speed)
    thermode.set_durations([40]*5) # 40 s
    # Ramp to the target temperature
    print(f"Ramping slowly to {target_temp}°C at {ramp_speed[0]}°C/s...")
    thermode.set_temperatures([target_temp]*5)  # Assumes this method performs a gradual ramp
    thermode.stimulate()

if __name__ == '__main__':
    main()