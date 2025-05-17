
"""
Functions that are helpful to run the slider
Slider: Sparrow 3x100 by MIDI Maker 
Author: Lucille Johnston
Updated: 3/20/25
"""

import mido
import time
from typing import Optional

class VASSlider:
    """
    A class to interact with a MIDI slider device (e.g., the Sparrow 3x100).
    
    This class provides methods to open a MIDI input port, extract slider positions from MIDI messages,
    and sample the slider continuously.
    """

    def __init__(self, port_name: Optional[str] = None):
        """
        Initializes the VASSlider instance by opening a MIDI input port.
        
        :param port_name: Name of the MIDI port (default: 'Sparrow 3x100' or first available if not found).
        """
        self.port_name = port_name or 'Sparrow 3x100'
        self.midi_port = self.open_midi_input(self.port_name)

    @staticmethod
    def open_midi_input(port_name: Optional[str] = None):
        """
        Opens a MIDI input port. If port_name is not specified or not found, the first available port is used.
        
        :param port_name: The MIDI port name.
        :return: An open MIDI input port.
        :raises RuntimeError: If no MIDI input ports are found.
        """
        ports = mido.get_input_names()
        if not ports:
            raise RuntimeError("No MIDI input ports found.")
        if port_name is None or port_name not in ports:
            port_name = ports[0]
        return mido.open_input(port_name)

    def get_value(self, side) -> Optional[int]:
        """
        Polls the MIDI input port and returns the current slider value.
        If a new control change message is received, it updates the value;
        otherwise, it returns the last known value (even if stationary).
        If no message has ever been received, returns 0.
        Changes which slider is sampled depending on which hand is moving the slider.
        """
        # Iterate over any pending messages without blocking.
        for message in self.midi_port.iter_pending():
            if side == 'left' and message.type == 'control_change' and message.control == 2:
                # Scale the MIDI value (0-127) to a 0-100 range.
                self.last_value = round((message.value / 127) * 100)
            elif side == 'right' and message.type == 'control_change' and message.control == 0:
                # Scale the MIDI value (0-127) to a 0-100 range.
                self.last_value = round((message.value / 127) * 100)
            elif side == 'both' and message.type == 'control_change':
                # Scale the MIDI value (0-127) to a 0-100 range.
                self.last_value = round((message.value / 127) * 100)
        # Initialize last_value to 0 if it doesn't exist.
        if not hasattr(self, 'last_value'):
            self.last_value = 0
        return self.last_value