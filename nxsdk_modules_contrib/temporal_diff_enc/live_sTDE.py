# INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
#
# Copyright © 2019-2021 Intel Corporation.
#
# This software and the related documents are Intel copyrighted
# materials, and your use of them is governed by the express
# license under which they were provided to you (License). Unless
# the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit  this software or the
# related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with
# no express or implied warranties, other than those that are
# expressly stated in the License.

"""
This tutorial shows how to use the DVS module.
The DVS module allows the live DVS hardware interface on Nahuku32 and Kapoho Bay.
This tutorial focuses on Kapoho Bay because it relies on Spike Output Ports
for live visualization (not supported on non-Kapoho Bay systems at the moment).

The DVS module also allows for playback of data from a .aedat file, and allows
a user to manually specify spike times and addresses. This functionality allows
a user to pipe recorded data through the processing pipeline for testing.

This tutorial covers live visualization of output spikes, different snips for
injecting DVS spikes (showing downsampling and flipping of spike co-ordinates),
and playback of spikes from an aedat file or custom user specified spikes.


NOTES

flashing squares only of one polarity, we need to understand how to connect and how the polarities are arranged into the visual field


commands to run loihi (getting_started_kb.html):
lsusb -t ---> check if your see the FTDI interfaces. If the driver says ftdi_sio,
you may remove these kernel modules using sudo rmmod ftdi_sio. Run lsusb -t to ensure that ftdi_sio is no longer there


"""

import nxsdk.api.n2a as nx
from nxsdk_modules.dvs.src.dvs import DVS
import numpy as np
import scipy.sparse as sps
import subprocess
import os
import errno
import inspect
from nxsdk_modules_contrib.temporal_diff_enc.stde import STDE_group


def startVisualizer(path, dimX=240, dimY=180, dimP=2, exec_name="/visualize_kb_spikes"):
    """Compiles and runs the DVS visualizer"""

    # compile the visualizer
    subprocess.run(["gcc"
                    + " -D DVS_X="
                    + str(dimX)
                    + " -D DVS_Y="
                    + str(dimY)
                    + " -D DVS_P="
                    + str(dimP)
                    + " -O3"
                    + " $(sdl2-config --cflags) "
                    + path
                    + " $(sdl2-config --libs)"
                    + " -o "
                    + os.path.dirname(path) + exec_name], shell=True)

    # setup spike fifo
    spikeFifoPath = os.path.dirname(path) + "/spikefifo"

    try:
        os.mkfifo(spikeFifoPath)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e

    # this environment variables sets where Loihi spikes will appear on the host
    os.environ['NX_SPIKE_OUTPUT'] = spikeFifoPath

    # run the visualizer
    subprocess.Popen(
        [os.path.dirname(path) + exec_name, "--source=" + spikeFifoPath, "--win_title=" + exec_name, "--interval=10",
         "--verbose=1"])


def setupNetwork(dimX=240, dimY=180, cp=None, stde_layer=False):
    """Sets up the basic DVS network which just repeats received spikes to
    a spike output port group for visualization.
    """
    net = nx.NxNet()
    # it allows a custom DVS injection
    dvs = DVS(net,
              dimX=dimX,
              dimY=dimY,
              dimP=2)
    # creating a prototype for a specific connection, excitatory in this case
    connproto = nx.ConnectionPrototype(weight=255, signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY)
    # creating a compartment
    if cp is None:
        cp = nx.CompartmentPrototype(vThMant=1,
                                     compartmentCurrentDecay=4095,
                                     compartmentVoltageDecay=4095)
    # creating a series of compartments of the same kind, cg maybe unnecessary
    cg = net.createCompartmentGroup(size=(dvs.numPixels), prototype=cp)
    # create network with compartments of the same kind
    dvs.outputs.rawDVS.connect(cg, prototype=connproto, connectionMask=sps.identity(dvs.numPixels))
    # dvs.outputs.rawDVS.connect(cg, prototype=connproto, connectionMask=sps.block_diag(np.arange(dvs.numPixels) % 2))
    # dvs.outputs.rawDVS.connect(cg, prototype=connproto, connectionMask=sps.block_diag(np.concatenate([np.ones(dvs.numPixels//2), np.zeros(dvs.numPixels//2)])))

    if stde_layer:
        params = {'tau_fac': 40,  # current tau of facilitator input
                  'tau_trigg': 40,  # current tau of trigger input
                  'tau_v': 40,  # voltage tau of TDE Neuron
                  'tau_c': 40,  # voltage tau of TDE Neuron
                  'weight_fac': 4000,  # amplitude of the facilitator spike
                  'do_probes': 'all',  # can be 'all', 'spikes' or None
                  'num_neurons': int(dvs.numPixels / 2),
                  }

        stde_group = STDE_group(params, net=net, dvs=dvs)

        # last_col_ids_pos = np.arange(-1, dvs.numPixels, dimX * dvs.pResolution, dtype=int)[1:]
        # last_col_ids_neg = np.arange(-2, dvs.numPixels, dimX * dvs.pResolution, dtype=int)[1:]
        # last_col = np.sort(np.concatenate((last_col_ids_pos, last_col_ids_neg)))
        #
        # first_col_ids_pos = np.arange(0, dvs.numPixels, dimX * dvs.pResolution, dtype=int)
        # first_col_ids_neg = np.arange(1, dvs.numPixels, dimX * dvs.pResolution, dtype=int)
        # first_col = np.sort(np.concatenate((first_col_ids_pos, first_col_ids_neg)))

        maskTrigger = sps.csc_matrix((dvs.numPixels // 2, dvs.numPixels))
        maskFacilitator = sps.csc_matrix((dvs.numPixels // 2, dvs.numPixels))

        # dvs pixels are organised by coloums with 2 polarities next to each other
        for i in range(int(dvs.numPixels / 2)):
            maskTrigger[i, 2 * i] = 1
            # maskTrigger[i, 2 * i +1] = 1
            try:
                maskFacilitator[i, 2 * (i + 1)] = 1
                # maskFacilitator[i, 2 * (i + 1)+1] = 1
            except IndexError:
                continue

        # maskTrigger[:, last_col] = 0
        # maskFacilitator[:, first_col] = 0

        dvs.outputs.rawDVS.connect(stde_group.neurongroup.dendrites[0].dendrites[0], prototype=connproto,
                                   connectionMask=maskTrigger)
        dvs.outputs.rawDVS.connect(stde_group.neurongroup.dendrites[0].dendrites[1], prototype=connproto,
                                   connectionMask=maskFacilitator)
    else:
        stde_group = None
    return net, cg, dvs, stde_group


def connectToVisualizer(outputCompartmentGroup):
    """Connects the output compartment group to the visualizer via a SpikeOutput PortGroup"""
    opg = outputCompartmentGroup.net.createSpikeOutputPortGroup(size=outputCompartmentGroup.numNodes)
    outputCompartmentGroup.connect(opg, connectionMask=sps.identity(outputCompartmentGroup.numNodes))


def runDVSandVisualize(visualizerPath=None,
                       snipFile=None,
                       funcName=None,
                       guardName=None,
                       dimX=240,
                       dimY=180,
                       aedatFilename=None,
                       customSpikes=None,
                       timesteps=10000):
    """Runs the DVS and visualizer"""
    visualiseFLAG = True
    stde_layer = False
    net, cg, dvs, stde_group = setupNetwork(dimX=dimX, dimY=dimY, stde_layer=stde_layer)

    # If no visualizerPath is given, omit the visualizer
    if visualizerPath is not None:
        startVisualizer(visualizerPath, dimX=dimX, dimY=dimY, exec_name='/visualize_stimulus')
        connectToVisualizer(cg) #stimulus
        # if stde_group is not None:
        #     startVisualizer(visualizerPath, dimX=dimX, dimY=dimY, dimP=2, exec_name='/visualize_stde')
        #     connectToVisualizer(stde_group.neurongroup.dendrites[0].dendrites[0])  # A
            # connectToVisualizer(stde_group.neurongroup.dendrites[0].dendrites[1]) # B

    compiler = nx.N2Compiler()
    board = compiler.compile(net)

    # If a filename was given, then use the file.
    # The DVS module will take care of communicating the file contents to the snip
    # File input takes precedence over the live input. Only one can be used at a time.
    if aedatFilename is not None:
        dvs.addFile(board, aedatFilename, 1000)

    # If spikes were manually specified, then use them.
    # The DVS module will take care of communicating the spike contents to the snip
    # This uses the same functionality as the aedat file input. An aedat file and
    # custom spikes can be mixed and will be superimposed.
    # Like the aedat file input, customSpikes take precedence over the live interface
    # and the two cannot be used simultaneously
    if customSpikes is not None:
        time = customSpikes[:, 0]
        x = customSpikes[:, 1]
        y = customSpikes[:, 2]
        p = customSpikes[:, 3]
        dvs.addSpikes(board, time, x, y, p)

    if snipFile is not None:
        # Specify the snip if a snip was given
        # Make sure to use the correct snip based on whether using the live interface
        # or a file/custom spikes as input
        dvs.setupSnips(board, snip=snipFile, funcName=funcName, guardName=guardName)
    else:
        # Otherwise dvs will take care of attaching the default snip.
        # The dvs module will choose the correct default snip based on whether we are
        # using a live interface or injecting spikes from the host (aedat file or manually specified)
        dvs.setupSnips(board)

    board.start()
    board.run(timesteps)
    board.disconnect()
    stde_group.plot()


if __name__ == "__main__":
    # Specify
    visualizerPath = os.path.dirname(inspect.getfile(startVisualizer)) + "/visualizer/default_visualizer.c"
    aedatFilename = os.path.dirname(inspect.getfile(startVisualizer)) + '/DAVIS240C_intel.aedat'

    full_resolution = True
    downsampled_resolution = False
    downsampled_manual = False
    inject_custom_spikes = False

    dimX = 240
    dimY = 180

    if full_resolution:
        print("Running Normal Full Resolution From File for 10 seconds")
        # Pass an input aedat file.
        # If a file is used, the default snip will switch to using the "dvs_host_spike_injection" function
        runDVSandVisualize(visualizerPath,
                           aedatFilename=aedatFilename)

        # Set a custom snip to downsample the input
        # This snip shows how downsampling can be done on the lakemont to save neural resources
        print("Running Downsampled Live for 10 seconds")
        snipFile = os.path.dirname(
            inspect.getfile(startVisualizer)) + "/snips/dvs_downsample.c"  # Note we use a different snip file here
        funcName = "dvs_live_spike_injection"
        guardName = "do_dvs_spike_injection"
        # The visualizer and dvs module need to know about the new dimensions to visualize correctly
        # and assign the correct number of compartments on Loihi
        runDVSandVisualize(visualizerPath,
                           snipFile=snipFile,
                           funcName=funcName,
                           guardName=guardName,
                           dimX=dimX,
                           dimY=dimY)

    if downsampled_resolution:
        # Set a custom snip to downsample the input, but using a file as input
        # The file input is useful for testing from recordings using the same pipeline
        # as the live interface
        print("Running Downsampled from file for 10 seconds")
        snipFile = os.path.dirname(inspect.getfile(startVisualizer)) + "/snips/dvs_downsample.c"
        funcName = "dvs_host_spike_injection"
        guardName = "do_dvs_spike_injection"
        # The visualizer and dvs module need to know about the new dimensions
        runDVSandVisualize(visualizerPath,
                           snipFile=snipFile,
                           funcName=funcName,
                           guardName=guardName,
                           dimX=dimX // 2,
                           dimY=dimY // 2,
                           aedatFilename=aedatFilename)

    if downsampled_manual:
        # Set a custom snip to downsample the input, but manually specify the input spikes
        print("Running Downsampled manual scan for 10 seconds")
        snipFile = os.path.dirname(inspect.getfile(startVisualizer)) + "/snips/dvs_downsample.c"
        funcName = "dvs_host_spike_injection"
        guardName = "do_dvs_spike_injection"
        # Manually specify spikes. Code below will make a diagonal line
        customSpikes = np.zeros((10000, 4), dtype=int)
        customSpikes[:, 0] = np.arange(10000)  # time in Loihi Timesteps
        customSpikes[:, 1] = np.arange(10000) % dimX  # x
        customSpikes[:, 2] = np.arange(10000) % dimY  # y
        customSpikes[:, 3] = np.arange(10000) % 2  # p
        # The visualizer and dvs module need to know about the new dimensions
        runDVSandVisualize(visualizerPath,
                           snipFile=snipFile,
                           funcName=funcName,
                           guardName=guardName,
                           dimX=dimX // 2,
                           dimY=dimY // 2,
                           customSpikes=customSpikes)

    if inject_custom_spikes:
        # Set a custom snip to downsample the input, but manually specify the input spikes
        print("Running Downsampled manual scan for 10 seconds")
        num_events = dimX * dimY
        snipFile = os.path.dirname(
            inspect.getfile(startVisualizer)) + "/snips/dvs_flipxy.c"  # uint32_t us timestep duration
        funcName = "dvs_host_spike_injection"
        guardName = "do_dvs_spike_injection"
        # Manually specify spikes. Code below will make a vertical line moving
        numFlashes = 50
        customSpikes = np.zeros((num_events * numFlashes, 4), dtype=int) # np.zeros((num_events, 4), dtype=int)
        customSpikes[:, 0] = np.repeat(np.arange(numFlashes), dimX * dimY) #(np.arange(num_events) // dimY)  # time in Loihi Timesteps
        customSpikes[:, 1] = np.tile(np.tile(np.arange(dimY), dimX), numFlashes) #np.arange(num_events) // dimY  # y
        customSpikes[:, 2] = np.tile(np.repeat(np.arange(dimX), dimY), numFlashes) #(np.arange(num_events) % dimY)  # x
        customSpikes[:, 3] = np.tile(np.repeat([0, 1], dimX * dimY), numFlashes // 2) # (np.arange(num_events) % 2)  # p
        # The visualizer and dvs module need to know about the new dimensions
        runDVSandVisualize(visualizerPath,
                           snipFile=snipFile,
                           funcName=funcName,
                           guardName=guardName,
                           dimX=dimX,
                           dimY=dimY,
                           customSpikes=customSpikes,
                           timesteps=customSpikes[-1, 0])
