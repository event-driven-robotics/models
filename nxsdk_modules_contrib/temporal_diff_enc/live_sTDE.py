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
"""

import nxsdk.api.n2a as nx
from nxsdk_modules.dvs.src.dvs import DVS
import numpy as np
import scipy.sparse as sps
import subprocess
import os
import errno
import inspect
from stde import STDE_group


def startVisualizer(path, dimX=240, dimY=180):
    """Compiles and runs the DVS visualizer"""

    # compile the visualizer
    subprocess.run(["gcc"
                    + " -D DVS_X="
                    + str(dimX)
                    + " -D DVS_Y="
                    + str(dimY)
                    + " -O3"
                    + " $(sdl2-config --cflags) "
                    + path
                    + " $(sdl2-config --libs)"
                    + " -o "
                    + os.path.dirname(path) + "/visualize_kb_spikes"], shell=True)

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
        [os.path.dirname(path) + "/visualize_kb_spikes", "--source=" + spikeFifoPath])


def setupNetwork(dimX=240, dimY=180, cp=None):
    """Sets up the basic DVS network which just repeats received spikes to
    a spike output port group for visualization.
    """
    net = nx.NxNet()
    dvs = DVS(net,
              dimX=dimX,
              dimY=dimY,
              dimP=2)

    connproto = nx.ConnectionPrototype(weight=255, signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY)

    if cp is None:
        cp = nx.CompartmentPrototype(vThMant=1,
                                     compartmentCurrentDecay=4095,
                                     compartmentVoltageDecay=4095)

    cg = net.createCompartmentGroup(size=(dvs.numPixels), prototype=cp)

    dvs.outputs.rawDVS.connect(cg, prototype=connproto, connectionMask=sps.identity(dvs.numPixels))

    params = {'tau_fac': 40,  # current tau of facilitator input
              'tau_trigg': 40,  # current tau of trigger input
              'tau_v': 40,  # voltage tau of TDE Neuron
              'tau_c': 40,  # voltage tau of TDE Neuron
              'weight_fac': 4000,  # amplitude of the facilitator spike
              'do_probes': None,  # can be 'all', 'spikes' or None
              'num_neurons': int(dvs.numPixels / 2),
              }

    stde_group = STDE_group(params, net = net)

    last_col_ids_pos = np.arange(-1, dvs.numPixels, dimX * dvs.pResolution, dtype=int)[1:]
    last_col_ids_neg = np.arange(-2, dvs.numPixels, dimX * dvs.pResolution, dtype=int)[1:]
    last_col = np.sort(np.concatenate((last_col_ids_pos, last_col_ids_neg)))

    first_col_ids_pos = np.arange(0, dvs.numPixels, dimX * dvs.pResolution, dtype=int)
    first_col_ids_neg = np.arange(1, dvs.numPixels, dimX * dvs.pResolution, dtype=int)
    first_col = np.sort(np.concatenate((first_col_ids_pos, first_col_ids_neg)))

    maskTrigger = sps.csc_matrix((int(dvs.numPixels / 2), dvs.numPixels))
    maskFacilitator = sps.csc_matrix((int(dvs.numPixels / 2), dvs.numPixels))

    for i in range(int(dvs.numPixels / 2)):
        maskTrigger[i, 2*i] = 1
        maskTrigger[i, 2*i + 1] = 1
        try:
            maskFacilitator[i, 2*(i+1)] = 1
            maskFacilitator[i, 2*(i+1) + 1] = 1
        except IndexError:
            continue

    maskTrigger[:, last_col] = 0
    maskFacilitator[:, first_col] = 0

    cg.connect(stde_group.neurongroup.dendrites[0].dendrites[0], prototype=connproto, connectionMask=maskTrigger)
    cg.connect(stde_group.neurongroup.dendrites[0].dendrites[1], prototype=connproto, connectionMask=maskFacilitator)


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
                       customSpikes=None):
    """Runs the DVS and visualizer"""

    net, cg, dvs, stde_group = setupNetwork(dimX=dimX, dimY=dimY)

    # If no visualizerPath is given, omit the visualizer
    if visualizerPath is not None:
        startVisualizer(visualizerPath, dimX=dimX, dimY=dimY)
        connectToVisualizer(stde_group.neurongroup.soma)

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
    board.run(10000)
    board.disconnect()
    stde_group.plot()


if __name__ == "__main__":
    # Specify
    visualizerPath = os.path.dirname(inspect.getfile(startVisualizer)) + "/visualizer/default_visualizer.c"
    aedatFilename = os.path.dirname(inspect.getfile(startVisualizer)) + '/DAVIS240C_intel.aedat'


    # print("Running Normal Full Resolution From File for 10 seconds")
    # # Pass an input aedat file.
    # # If a file is used, the default snip will switch to using the "dvs_host_spike_injection" function
    # runDVSandVisualize(visualizerPath,
    #                    aedatFilename=aedatFilename)
    #
    # # Set a custom snip to downsample the input
    # # This snip shows how downsampling can be done on the lakemont to save neural resources
    # print("Running Downsampled Live for 10 seconds")
    # snipFile = os.path.dirname(
    #     inspect.getfile(startVisualizer)) + "/snips/dvs_downsample.c"  # Note we use a different snip file here
    # funcName = "dvs_live_spike_injection"
    # guardName = "do_dvs_spike_injection"
    # # The visualizer and dvs module need to know about the new dimensions to visualize correctly
    # # and assign the correct number of compartments on Loihi
    # runDVSandVisualize(visualizerPath,
    #                    snipFile=snipFile,
    #                    funcName=funcName,
    #                    guardName=guardName,
    #                    dimX=120,
    #                    dimY=90)

    # # Set a custom snip to downsample the input, but using a file as input
    # # The file input is useful for testing from recordings using the same pipeline
    # # as the live interface
    # print("Running Downsampled from file for 10 seconds")
    # snipFile = os.path.dirname(inspect.getfile(startVisualizer)) + "/snips/dvs_downsample.c"
    # funcName = "dvs_host_spike_injection"
    # guardName = "do_dvs_spike_injection"
    # # The visualizer and dvs module need to know about the new dimensions
    # runDVSandVisualize(visualizerPath,
    #                   snipFile=snipFile,
    #                    funcName=funcName,
    #                    guardName=guardName,
    #                    dimX=120,
    #                    dimY=90,
    #                    aedatFilename=aedatFilename)

    # # Set a custom snip to downsample the input, but manually specify the input spikes
    # print("Running Downsampled manual scan for 10 seconds")
    # snipFile = os.path.dirname(inspect.getfile(startVisualizer)) + "/snips/dvs_downsample.c"
    # funcName = "dvs_host_spike_injection"
    # guardName = "do_dvs_spike_injection"
    # # Manually specify spikes. Code below will make a diagonal line
    # customSpikes = np.zeros((10000, 4), dtype=int)
    # customSpikes[:, 0] = np.arange(10000)  # time in Loihi Timesteps
    # customSpikes[:, 1] = np.arange(10000) % 240  # x
    # customSpikes[:, 2] = np.arange(10000) % 180  # y
    # customSpikes[:, 3] = np.arange(10000) % 2  # p
    # # The visualizer and dvs module need to know about the new dimensions
    # runDVSandVisualize(visualizerPath,
    #                    snipFile=snipFile,
    #                    funcName=funcName,
    #                    guardName=guardName,
    #                    dimX=120,
    #                    dimY=90,
    #                    customSpikes=customSpikes)

    # Set a custom snip to downsample the input, but manually specify the input spikes
    print("Running Downsampled manual scan for 10 seconds")
    timesteps = 43200
    snipFile = os.path.dirname(inspect.getfile(startVisualizer)) + "/snips/dvs_downsample.c"
    funcName = "dvs_host_spike_injection"
    guardName = "do_dvs_spike_injection"
    # Manually specify spikes. Code below will make a vertocal  line moving
    customSpikes = np.zeros((timesteps, 4), dtype=int)
    customSpikes[:, 0] = (np.arange(timesteps) // 180) * 10 # time in Loihi Timesteps
    customSpikes[:, 1] = np.arange(timesteps) // 180  # x
    customSpikes[:, 2] = np.arange(timesteps) % 180  # y
    customSpikes[:, 3] = np.arange(timesteps) % 2  # p
    # The visualizer and dvs module need to know about the new dimensions
    runDVSandVisualize(visualizerPath,
                       snipFile=snipFile,
                       funcName=funcName,
                       guardName=guardName,
                       dimX=120,
                       dimY=90,
                       customSpikes=customSpikes)

