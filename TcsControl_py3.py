#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=== Authors ===
Dr. Ulrike Horn: uhorn@cbs.mpg.de
Max Planck Institute for Human Cognitive and Brain Sciences
Research group Pain Perception
Date: 26th May 2021
"""

import serial  # Import the serial module to communicate with hardware via serial ports
import time 

class TcsDevice:
    """
    A class to communicate with a thermode control system (TCS) via a serial connection.
    This class sends commands to control temperature settings and retrieve temperature readings.
    """

    def __init__(self, port='/dev/ttyACM0'):
        """
        Initializes the TCS device by setting up the serial connection.
        
        :param port: The serial port to which the device is connected (default: '/dev/ttyACM0').
        """
        self.baseline = 30.0  # Default baseline temperature (neutral temperature)

        # Open serial communication with the thermode device
        self.s_port = serial.Serial(port, baudrate=115200, timeout=2)
        self.s_port.flushInput()  # Clear any existing input data from the serial buffer
        
        # Send an initial handshake command ('H') to establish communication
        self.s_port.write(bytes(b'H'))
        self.s_port.flushOutput()  # Ensure data is fully sent

        # Read firmware version and device ID from the thermode
        firmware_msg = self.s_port.read(30)  # Read up to 30 bytes from the device
        print(firmware_msg)  # Print the firmware message for debugging
        
        self.s_port.flushInput()  # Clear buffer before reading the ID message
        id_msg = self.s_port.read(30)  # Read device identification message
        print(id_msg)  # Print ID message for debugging
        
        self.s_port.flushInput()  # Clear input buffer
        rest = self.s_port.read(10000)  # Read any remaining data (if any)
        self.s_port.flushInput()  # Ensure buffer is empty

    def set_quiet(self):
        """
        Sets the thermode to quiet mode.
        This prevents the device from continuously sending temperature data (1Hz idle, 100Hz during stimulation),
        which can interfere with commands from the PC.
        """
        self.s_port.write(bytes(b'F'))  # Send 'F' command to enable quiet mode
        self.s_port.flushOutput()

    def set_baseline(self, baselineTemp):
        """
        Sets the baseline (neutral) temperature.
        
        :param baselineTemp: A float value representing the baseline temperature (min: 20°C, max: 40°C).
        """
        # Ensure temperature is within the allowed range
        if baselineTemp > 40:
            baselineTemp = 40
        if baselineTemp < 20:
            baselineTemp = 20

        # Format the command and send it to the device
        command = b'N%03d' % (baselineTemp * 10)
        self.s_port.write(bytes(command))
        self.s_port.flushOutput()

    def set_durations(self, stimDurations):
        """
        Sets the stimulus durations for all 5 zones.
        
        :param stimDurations: A list of 5 values (each between 0.001s and 99.999s).
        """
        for i in range(5):
            if stimDurations[i] > 99.999:
                stimDurations[i] = 99.999
            if stimDurations[i] < 0.001:
                stimDurations[i] = 0.001

        # If all durations are the same, send one command for all zones
        if stimDurations.count(stimDurations[0]) == len(stimDurations):
            command = b'D0%05d' % (stimDurations[1] * 1000)
            self.s_port.write(bytes(command))
            self.s_port.flushOutput()
        else:
            # Send individual commands for each zone
            for i in range(5):
                command = b'D%d%05d' % ((i + 1), (stimDurations[i] * 1000))
                self.s_port.write(bytes(command))
                self.s_port.flushOutput()

    def set_ramp_speed(self, rampSpeeds):
        """
        Sets the ramp-up speed (how quickly the temperature increases).
        
        :param rampSpeeds: A list of 5 values (each between 0.1°C/s and 300°C/s).
        """
        for i in range(5):
            if rampSpeeds[i] > 300:
                rampSpeeds[i] = 300
            if rampSpeeds[i] < 0.1:
                rampSpeeds[i] = 0.1

        # If all speeds are the same, send one command for all zones
        if rampSpeeds.count(rampSpeeds[0]) == len(rampSpeeds):
            command = b'V0%04d' % (rampSpeeds[1] * 10)
            self.s_port.write(bytes(command))
            self.s_port.flushOutput()
        else:
            # Send individual commands for each zone
            for i in range(5):
                command = b'V%d%04d' % ((i + 1), (rampSpeeds[i] * 10))
                self.s_port.write(bytes(command))
                self.s_port.flushOutput()

    def set_return_speed(self, returnSpeeds):
        """
        Sets the ramp-down speed (how quickly the temperature decreases).
        
        :param returnSpeeds: A list of 5 values (each between 0.1°C/s and 300°C/s).
        """
        for i in range(5):
            if returnSpeeds[i] > 300:
                returnSpeeds[i] = 300
            if returnSpeeds[i] < 0.1:
                returnSpeeds[i] = 0.1

        if returnSpeeds.count(returnSpeeds[0]) == len(returnSpeeds):
            command = b'R0%04d' % (returnSpeeds[1] * 10)
            self.s_port.write(bytes(command))
            self.s_port.flushOutput()
        else:
            for i in range(5):
                command = b'R%d%04d' % ((i + 1), (returnSpeeds[i] * 10))
                self.s_port.write(bytes(command))
                self.s_port.flushOutput()

    def set_temperatures(self, temperatures):
        """
        Sets the target temperatures for all 5 zones.
        
        :param temperatures: A list of 5 values (each between 0.1°C and 60°C).
        """
        for i in range(5):
            if temperatures[i] > 60:
                temperatures[i] = 60
            if temperatures[i] < 0.1:
                temperatures[i] = 0.1

        if temperatures.count(temperatures[0]) == len(temperatures):
            command = b'C0%03d' % (temperatures[1] * 10)
            self.s_port.write(bytes(command))
            self.s_port.flushOutput()
        else:
            for i in range(5):
                command = b'C%d%03d' % ((i + 1), (temperatures[i] * 10))
                self.s_port.write(bytes(command))
                self.s_port.flushOutput()

    def stimulate(self):
        """
        Starts the stimulation process using the previously set parameters.
        """
        self.s_port.write(bytes(b'L'))

    def get_temperatures(self):
        """
        Retrieves the current temperatures of all five zones.
        
        :return: A list of 5 temperature values or an empty list if an error occurs.
        """
        self.s_port.flushInput()
        self.s_port.write(bytes(b'E'))  # Request temperature readings
        self.s_port.flushOutput()

        data = self.s_port.read(24)  # Read the temperature data from the device
        temperatures = [0, 0, 0, 0, 0]

        if len(data) > 23:
            temperatures[0] = float(data[5:8]) / 10
            temperatures[1] = float(data[9:12]) / 10
            temperatures[2] = float(data[13:16]) / 10
            temperatures[3] = float(data[17:20]) / 10
            temperatures[4] = float(data[21:24]) / 10
        else:
            temperatures = []  # Return empty list if data is invalid

        return temperatures

    def close(self):
        """
        Closes the serial connection to the device.
        """
        self.s_port.close()

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