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
from scipy import interpolate as interp

# This is for writing .wav file
from scipy.io.wavfile import write


try:
    from console_examples_util import config_first_detected_device
except ImportError:
    from .console_examples_util import config_first_detected_device

def xcorr(x, y):
    # factor = 4
    # print("here")
    # # first interpolate the signals so that there are more data points (higher resolution)
    # fx = interp.interp1d(np.arange(0,x.shape[1]),x,'quadratic')
    # x = fx(np.arange(0,factor*x.shape[1]))
    # y = interp.interp1d(y.shape[1],factor*y.shape[1],'quadratic')
    # print("now here")
    # returns actual cross-correlation array
    corr = signal.correlate(x, y, mode="full")
    # returns the array of time differences
    lags = signal.correlation_lags(x.size, y.size, mode="full")
    # return the TDOA fo print(lags.shape) in samples
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
    T = 5
    window = int(rate * T)
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
        tdoa(np_data, num_chans, T)

    except Exception as e:
        print('\n', e)
    finally:
        if memhandle:
            # Free the buffer in a finally block to prevent a memory leak.
            ul.win_buf_free(memhandle)
        if use_device_detection:
            ul.release_daq_device(board_num)

def tdoa(np_data, num_chans, T):
    rate = 44100
    # plt.plot(np_data.transpose())
    # plt.show()

    TDOA = np.empty((num_chans,num_chans))

    for i in range(num_chans):
        for j in range(i+1, num_chans):
            # print(f"Channel {i}")
            # print(f"Channel {j}")
            [tdoa, lags, conv_data] = xcorr(np_data[i].flat[:], np_data[j].flat[:])
            # plt.plot(lags, conv_data.transpose())
            # plt.show()
            print(tdoa)
            TDOA[i,j] = tdoa/rate
            TDOA[j,i] = -(tdoa/rate)
    
    print(TDOA)

    # make distances matrix
    distances = np.array([0, 15, 30, 45])
    one = np.ones([distances.size, distances.size])
    distances = distances * one 
    distances = np.abs(distances - distances.transpose())/100   # have in meters
    print(distances)


    # Calculate angle of arrival:
    c = 343;    # speed of sound in air
    y = (c*TDOA/distances)

    theta = np.arcsin(y) * 180 / np.pi  # in degrees

    # Store the signs from angle of arrival to identify which side source is from -- don't 
    signs = np.copy(theta)
    signs[signs > 0] = 1
    signs[signs < 0] = -1


    # Fix the signs of the AOAs to get uniform *direction* of arrival
    for i in range(len(theta)):
        for j in range(len(theta[i])):
            if j < i:
                theta[i][j]=(-1) * theta[i][j]


    # Now average the AOA as seen from each microphone element
    avg_theta = np.empty(np.shape(y)[0])
    for index, row in enumerate(theta):
        sum = 0
        items = 0
        for elem in row:
            if not np.isnan(elem) and not np.isinf(elem):
                sum += elem
                items += 1
        
        avg_theta[index] = sum/items

    print("The AOA vector is: {}".format(avg_theta))

    # PLOT THE AOA as a line-of-sight from array
    # unpack the first point
    fig, ax1 = plt.subplots() 

    # save the slope and intercepts
    M = np.ones((4,2)) * -1
    b = np.ones((4,1))
    
    line_length = 1000  # 10 meters
    for idx, x in enumerate(range(-23,23,15)):

        y = 0

        # find the end point
        endy = y + line_length * np.sin(np.radians(avg_theta[3-idx]+90))
        endx = x + line_length * np.cos(np.radians(avg_theta[3-idx]+90))
        # print("x at {}, is: {}".format(idx,x))
        
        # plot the points
        ax1.plot([x, endx], [y, endy])
        # ax2.plot([x, endx], [y, endy])
        m = (endy-y)/(endx-x)
        M[idx,0] = m
        b[idx] = (y-m*x)*-1

    # add twin axis so we can show left and right labels
    ax2 = ax1.twinx()
    ax1.set_xlabel('Base of Array (cm)')    
    ax1.set_ylabel('Left')
    ax2.set_ylabel('Right')

    # plot basic cardinals from center:
    x = 0
    end45 = [x+line_length * np.cos(np.radians(45)), y + line_length*np.sin(np.radians(45))]
    end45neg = [x+line_length * np.cos(np.radians(45 + 90)), y + line_length*np.sin(np.radians(90+45))]
    end90 = [x+line_length * np.cos(np.radians(90)), y + line_length*np.sin(np.radians(90))]
    ax1.plot([x,end45[0]],[y,end45[1]],'k--')
    ax1.plot([x,end45neg[0]],[y,end45neg[1]],'k--')
    ax1.plot([x,end90[0]],[y,end90[1]],'k--')

    # set bounds and show
    plt.ylim([-5, line_length]) 
    plt.xlim([-line_length, line_length])
    plt.title("Microphone Array AOA View")
    plt.show()
    #print("before", file=sys.stderr)
    #print("after", file=sys.stderr)

    # calculate intersection
    print(M)
    M_inv = np.linalg.pinv(M)
    intersection = np.matmul(M_inv, b)

    r = np.sqrt(intersection[0]**2 + intersection[1]**2)
    angle = (np.arctan2(intersection[0],intersection[1]) * 180 /np.pi) + 90
    return r, angle
       



if __name__ == '__main__':
    run_example()
