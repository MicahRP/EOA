from __future__ import absolute_import, division, print_function
from builtins import *  # @UnusedWildImport
import sys

from ctypes import cast, POINTER, c_double

from mcculw import ul
from mcculw.enums import ScanOptions, ChannelType, ULRange, DigitalPortType
from mcculw.device_info import DaqDeviceInfo
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# This is for writing .wav file
from scipy.io.wavfile import write


try:
    from console_examples_util import config_first_detected_device
except ImportError:
    from .console_examples_util import config_first_detected_device

def xcorr(x, y):
    # returns actual cross-correlation array
    corr = signal.correlate(x, y, mode="full")
    # returns the array of time differences
    lags = signal.correlation_lags(x.size, y.size, mode="full")
    # return the TDOA fo print(lags.shape)
    tdoa = lags[np.argmax(corr)]
    return tdoa, lags, corr

def run_example():
    # By default, the example detects and displays all available devices and
    # selects the first device listed. Use the dev_id_list variable to filter
    # detected devices by device ID (see UL documentation for device IDs).
    # If use_device_detection is set to False, the board_num variable needs to
    # match the desired board number configured with Instacal.
    use_device_detection = True
    dev_id_list = [317, 318]  # USB-1808 = 317, USB-1808X = 318
    board_num = 0
    # Supported PIDs for the USB-1808 Series
    rate = 44100
    T = 1
    points_per_channel = rate * T
    memhandle = None

    try:
        if use_device_detection:
            config_first_detected_device(board_num, dev_id_list)

        daq_dev_info = DaqDeviceInfo(board_num)
        print('\nActive DAQ device: ', daq_dev_info.product_name, ' (',
              daq_dev_info.unique_id, ')\n', sep='')

        scan_options = ScanOptions.FOREGROUND | ScanOptions.SCALEDATA

        # Create the daq_in_scan channel configuration lists
        chan_list = []
        chan_type_list = []
        gain_list = []
        all_data = []

        # Analog channels must be first in the list
        
        chan_list.append(0)
        chan_type_list.append(ChannelType.ANALOG_DIFF)
        gain_list.append(ULRange.BIP10VOLTS)
        all_data.append([])
        chan_list.append(1)
        chan_type_list.append(ChannelType.ANALOG_DIFF)
        gain_list.append(ULRange.BIP10VOLTS)
        all_data.append([])
        chan_list.append(2)
        chan_type_list.append(ChannelType.ANALOG_DIFF)
        gain_list.append(ULRange.BIP10VOLTS)
        all_data.append([])
        chan_list.append(3)
        chan_type_list.append(ChannelType.ANALOG_DIFF)
        gain_list.append(ULRange.BIP10VOLTS)
        all_data.append([])

        num_chans = len(chan_list)

        total_count = num_chans * points_per_channel

        # Allocate memory for the scan and cast it to a ctypes array pointer
        memhandle = ul.scaled_win_buf_alloc(total_count)
        ctypes_array = cast(memhandle, POINTER(c_double))

        # Note: the ctypes array will no longer be valid after win_buf_free is
        # called.
        # A copy of the buffer can be created using win_buf_to_array or
        # win_buf_to_array_32 before the memory is freed. The copy can be used
        # at any time.

        # Check if the buffer was successfully allocated
        if not memhandle:
            raise Exception('Error: Failed to allocate memory')

        # Start the scan
        ul.daq_in_scan(
            board_num, chan_list, chan_type_list, gain_list, num_chans,
            rate, 0, total_count, memhandle, scan_options)

        print('Scan completed successfully. Data:')

        # Create a format string that aligns the data in columns
        row_format = '{:>5}' + '{:>10}' * num_chans

        # Print the channel name headers
        labels = ['Index']
        for ch_index in range(num_chans):
            channel_label = {
                ChannelType.ANALOG: lambda:
                    'AI' + str(chan_list[ch_index]),
                ChannelType.ANALOG_DIFF: lambda:
                    'AI' + str(chan_list[ch_index]),
                ChannelType.ANALOG_SE: lambda:
                    'AI' + str(chan_list[ch_index]),
                ChannelType.DIGITAL: lambda:
                    chan_list[ch_index].name,
                ChannelType.CTR: lambda:
                    'CI' + str(chan_list[ch_index]),
            }[chan_type_list[ch_index]]()
            labels.append(channel_label)
        print(row_format.format(*labels))

        # Print the data
        data_index = 0
        for index in range(points_per_channel):
            display_data = [index]
            for ch_index in range(num_chans):
                data_label = {
                    ChannelType.ANALOG: lambda:
                        '{:.3f}'.format(ctypes_array[data_index]),
                    ChannelType.ANALOG_DIFF: lambda:
                        '{:.3f}'.format(ctypes_array[data_index]),
                    ChannelType.ANALOG_SE: lambda:
                        '{:.3f}'.format(ctypes_array[data_index]),
                    ChannelType.DIGITAL: lambda:
                        '{:d}'.format(int(ctypes_array[data_index])),
                    ChannelType.CTR: lambda:
                        '{:d}'.format(int(ctypes_array[data_index])),
                }[chan_type_list[ch_index]]()

                # display_data.append(data_label)
                all_data[ch_index].append(ctypes_array[data_index])
                data_index += 1
            # Print this row
            # print(row_format.format(*display_data))
        
        np_data = np.empty((num_chans, points_per_channel))
        for channel_idx in range(num_chans):
            np_data[channel_idx] = np.asarray(all_data[channel_idx])


       # convert to matrix
        np_data = np.mat(np_data)

        # de-mean data:
        np_data = np_data - np.mean(np_data,axis=1)

        plt.plot(np_data.transpose())
        plt.show()

        TDOA = np.empty((num_chans,num_chans))

        for i in range(num_chans):
            for j in range(i+1, num_chans):
                # print(f"Channel {i}")
                # print(f"Channel {j}")
                [tdoa, lags, conv_data] = xcorr(np_data[i],np_data[j])
                # plt.plot(lags, conv_data.transpose())
                # plt.show()
                TDOA[i,j] = tdoa
                TDOA[j,i] = tdoa
        
        TDOA[TDOA < 1] = 0 

        # make distances matrix
        distances = np.array([0, 15, 30, 45])
        one = np.ones([distances.size, distances.size])
        distances = distances * one 
        distances = np.abs(distances - distances.transpose())/100   # have in meters
        # print(distances)


        ### NEED TO GET THIS PART ###
        # Calculate angle of arrival:
        c = 346;    # speed of sound in air

        y = (c*TDOA/distances)
        
        print(y)
        theta = np.arcsin(y)
        print(theta)

        #print("before", file=sys.stderr)
       # print("after", file=sys.stderr)




    except Exception as e:
        print('\n', e)
    finally:
        if memhandle:
            # Free the buffer in a finally block to prevent a memory leak.
            ul.win_buf_free(memhandle)
        if use_device_detection:
            ul.release_daq_device(board_num)


if __name__ == '__main__':
    run_example()
