/*
INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY

Copyright © 2019-2021 Intel Corporation.

This software and the related documents are Intel copyrighted
materials, and your use of them is governed by the express
license under which they were provided to you (License). Unless
the License provides otherwise, you may not use, modify, copy,
publish, distribute, disclose or transmit  this software or the
related documents without Intel's prior written permission.

This software and the related documents are provided as is, with
no express or implied warranties, other than those that are
expressly stated in the License.
*/

#include "dvs_flipxy.h"

static int channelID = -1;
static int index = 0;
static DvsFileSpike file_data[DVS_FILE_SPIKES_PER_MESSAGE];

int do_dvs_spike_injection(runState *s) {

    if(s->time_step==1) {
      channelID = getChannelID(DVS_FILE_RECEIVE_NAME);
      if(channelID == -1) {
          channelID = getChannelID(DVS_LIVE_RECEIVE_NAME);
          if(channelID == -1)
              printf("Invalid channelID for nxdvs_data\n");
      } else
          readChannel(channelID,&file_data,1);
    }
    return 1;
}

// handles receiving dvs events from a live sensor
void dvs_live_spike_injection(runState *s) {
    uint32_t us = 1000; //controls Loihi's timestep duration
    uint64_t durationTicks = us * TICKS_PER_MICROSECOND;
    uint64_t now = timestamp();
    uint64_t deadline = now + durationTicks;

    DvsData data;

    while(now < deadline) {
        uint8_t avail = probeChannel(channelID);
        for (int ii = 0; ii < avail; ++ii)  {
            readChannel(channelID,&data,1);
            for (int jj = 0; jj < DVS_LIVE_SPIKES_PER_MESSAGE; ++jj) {
                uint8_t x = data.spikes[jj].x;
                uint8_t y = data.spikes[jj].y;
                uint8_t p = (data.polarity & 1);
                data.polarity >>= 1;
                inject_dvs_spike(x, y, p, s->time_step);
            }
         }
     now = timestamp();
    }
}

// handles receiving dvs events from the host
void dvs_host_spike_injection(runState *s) {
    //optionally slow Loihi down for real-time visualization
    uint32_t us = 1000000;
    uint64_t durationTicks = us * TICKS_PER_MICROSECOND;
    uint64_t now = timestamp();
    uint64_t deadline = now + durationTicks;
    
    uint8_t this_timestep = s->time_step & 0xff;
    while (this_timestep == file_data[index].time) {
        inject_dvs_spike(file_data[index].x, file_data[index].y, file_data[index].p, file_data[index].time);
        index++;
        if (index == DVS_FILE_SPIKES_PER_MESSAGE) {
            readChannel(channelID,&file_data,1);
            index = 0;
        }
    }
    
    while(timestamp() < deadline);
}

// Handles translating x,y,p,t to chip, core, axon for both the live and host/file interface
void inject_dvs_spike(uint8_t x, uint8_t y, uint8_t p, uint8_t t) {
    if (x < WIDTH && y < HEIGHT) {    // coordinates out of expected range!
        // compute core/axon
        uint32_t index = (WIDTH-x-1)*(HEIGHT*2)+(HEIGHT-y-1)*2+p; //flip x and y
        uint32_t logicalCore = index/COMPARTMENTS_PER_CORE;
                     
        CoreId core = nx_nth_coreid(logicalCore);
        uint32_t axon  = index % COMPARTMENTS_PER_CORE;

        //send spike
        nx_send_discrete_spike(t, core, axon);
        printf("snip received event at x=%d, y=%d, p=%d, time=%d\n", x,y,p,t);
        printf("snip injecting spikes to core %d, neuron %d\n", logicalCore, axon);


    }
}


