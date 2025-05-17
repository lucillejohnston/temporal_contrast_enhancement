import serial


class eVASSlider:
    def __init__(self, port, baud_rate = 115200):
        self.port = port
        self.ser = self.open_serial_port(port, baud_rate)
        self.last_value = None
    
    def open_serial_port(self, port, baud_rate):
        """Connect to the eVAS slider with the specified parameters."""
        try:
            slider_ser = serial.Serial(
                port,
                baudrate=baud_rate,
                bytesize=serial.EIGHTBITS,    # 8 data bits
                parity=serial.PARITY_NONE,    # No parity
                stopbits=serial.STOPBITS_ONE, # 1 stop bit
                timeout=1
            )
        except serial.SerialException as e:
            print("Error opening serial port:", e)
            exit()
        print("eVAS Slider connected on", port)
        return slider_ser
    
    def get_value(self):
        """
        Read and return the slider value from the serial port.
        Prints the slider value if numeric or prints debug info if non-numeric.
        Converts from a 0-999 range to a 0-100 scale
        """
        raw_data = self.ser.readline()
        try:
            data = raw_data.decode('utf-8').rstrip()
            if data:
                try:
                    value = float(data)
                    adjusted_value = int(round((value / 999) * 100))
                    self.last_value = adjusted_value 
                    # print("Slider value:", slider_value)
                except ValueError:
                    # print("Received non-numeric data:", data)
                    pass
        except UnicodeDecodeError:
            pass
        if self.last_value is None:
            self.last_value = 0
        return self.last_value
    

# ----------------------------------------
# For testing purposes 
# change port as needed 
# ----------------------------------------
if __name__ == "__main__":
    slider = eVASSlider(port='COM7') 
    while True:
        value = slider.get_value()
        print("Slider value:", value)