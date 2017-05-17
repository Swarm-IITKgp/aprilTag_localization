/* Copyright (C) 2013-2016, The Regents of The University of Michigan.
All rights reserved.

This software was developed in the APRIL Robotics Lab under the
direction of Edwin Olson, ebolson@umich.edu. This software may be
available under alternative licensing terms; contact the address above.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the Regents of The University of Michigan.
*/

#include <iostream>
#include <cmath>
#include "opencv2/opencv.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <string>
#include <thread>
#include <ignition/msgs.hh>
#include <ignition/transport.hh>

#include "apriltag/apriltag.h"
#include "apriltag/tag36h11.h"
#include "apriltag/tag36h10.h"
#include "apriltag/tag36artoolkit.h"
#include "apriltag/tag25h9.h"
#include "apriltag/tag25h7.h"
#include "apriltag/common/getopt.h"

#define BID1 1
#define BID2 12
#define BID3 3

using namespace std;
using namespace cv;

static std::atomic<bool> g_terminatePub(false);

void signal_handler(int _signal)
{
  if (_signal == SIGINT || _signal == SIGTERM)
    g_terminatePub = true;
}

int main(int argc, char *argv[])
{

	// Install a signal handler for SIGINT and SIGTERM.
	std::signal(SIGINT,  signal_handler);
	std::signal(SIGTERM, signal_handler);
	
	ignition::transport::Node node;
	std::string topic1 = "/bot1/pose";
	std::string topic2 = "/bot2/pose";
	std::string topic3 = "/bot3/pose";

	auto pub1 = node.Advertise<ignition::msgs::Pose>(topic1);
	auto pub2 = node.Advertise<ignition::msgs::Pose>(topic2);
	auto pub3 = node.Advertise<ignition::msgs::Pose>(topic3);

    getopt_t *getopt = getopt_create();

    getopt_add_bool(getopt, 'h', "help", 0, "Show this help");
    getopt_add_bool(getopt, 'd', "debug", 0, "Enable debugging output (slow)");
    getopt_add_bool(getopt, 'q', "quiet", 0, "Reduce output");
    getopt_add_string(getopt, 'f', "family", "tag36h11", "Tag family to use");
    getopt_add_int(getopt, '\0', "border", "1", "Set tag family border size");
    getopt_add_int(getopt, 't', "threads", "4", "Use this many CPU threads");
    getopt_add_double(getopt, 'x', "decimate", "1.0", "Decimate input image by this factor");
    getopt_add_double(getopt, 'b', "blur", "0.0", "Apply low-pass blur to input");
    getopt_add_bool(getopt, '0', "refine-edges", 1, "Spend more time trying to align edges of tags");
    getopt_add_bool(getopt, '1', "refine-decode", 0, "Spend more time trying to decode tags");
    getopt_add_bool(getopt, '2', "refine-pose", 0, "Spend more time trying to precisely localize tags");

    if (!getopt_parse(getopt, argc, argv, 1) ||
            getopt_get_bool(getopt, "help")) {
        printf("Usage: %s [options]\n", argv[0]);
        getopt_do_usage(getopt);
        exit(0);
    }

	// Prepare the message.
	ignition::msgs::Pose msg;
	ignition::msgs::Quaternion quat;
	ignition::msgs::Vector3d vec;
	msg.set_allocated_position(&vec);
	msg.set_allocated_orientation(&quat);
	double x = 0, y = 0, z = 0;
	double xq = 0, yq = 0, zq = 0, wq = 0;
    // Initialize camera
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Couldn't open video capture device" << endl;
        return -1;
    }

    // Initialize tag detector with options
    apriltag_family_t *tf = NULL;
    const char *famname = getopt_get_string(getopt, "family");
    if (!strcmp(famname, "tag36h11"))
        tf = tag36h11_create();
    else if (!strcmp(famname, "tag36h10"))
        tf = tag36h10_create();
    else if (!strcmp(famname, "tag36artoolkit"))
        tf = tag36artoolkit_create();
    else if (!strcmp(famname, "tag25h9"))
        tf = tag25h9_create();
    else if (!strcmp(famname, "tag25h7"))
        tf = tag25h7_create();
    else {
        printf("Unrecognized tag family name. Use e.g. \"tag36h11\".\n");
        exit(-1);
    }
    tf->black_border = getopt_get_int(getopt, "border");

    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_detector_add_family(td, tf);
    td->quad_decimate = getopt_get_double(getopt, "decimate");
    td->quad_sigma = getopt_get_double(getopt, "blur");
    td->nthreads = getopt_get_int(getopt, "threads");
    td->debug = getopt_get_bool(getopt, "debug");
    td->refine_edges = getopt_get_bool(getopt, "refine-edges");
    td->refine_decode = getopt_get_bool(getopt, "refine-decode");
    td->refine_pose = getopt_get_bool(getopt, "refine-pose");

    Mat frame, gray;
    while (!g_terminatePub) {
        cap >> frame;
	 // for checking the speed
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Make an image_u8_t header for the Mat data
        image_u8_t im = { .width = gray.cols,
            .height = gray.rows,
            .stride = gray.cols,
            .buf = gray.data
        };
	//double tick = (double)getTickCount();
        zarray_t *detections = apriltag_detector_detect(td, &im);
	//AvrgTime.first += ((double)getTickCount() - tick) / getTickFrequency();
	//AvrgTime.second++;
	//cout << "\rTime detection=" << 1000 * AvrgTime.first / AvrgTime.second << " ";
        cout << zarray_size(detections) << " tags detected" << endl;

        // Draw detection outlines
        for (int i = 0; i < zarray_size(detections); i++) {
            apriltag_detection_t *det;
            zarray_get(detections, i, &det);
			int cx = (det->p[0][0] + det->p[1][0] + det->p[2][0] + det->p[3][0])/4;
			int cy = (det->p[0][1] + det->p[1][1] + det->p[2][1] + det->p[3][1])/4;
			float angle = 180*atan2((det->p[0][1] + det->p[1][1])/2 - (det->p[3][1] + det->p[2][1])/2, (det->p[0][0] + det->p[1][0])/2 - (det->p[3][0] + det->p[2][0])/2)/3.14;
			if(det->id == BID1){
				vec.set_x(cx);
				vec.set_y(cy);
				vec.set_z(0);
				quat.set_x(0);
				quat.set_y(0);
				quat.set_z(angle);
				quat.set_w(0);
				if (!pub1.Publish(msg))
				  break;
				std::cout << "Publishing hello on topic [" << topic1 << "]" << std::endl;
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
			}
			if(det->id == BID2){
				vec.set_x(cx);
				vec.set_y(cy);
				vec.set_z(0);
				quat.set_x(0);
				quat.set_y(0);
				quat.set_z(angle);
				quat.set_w(0);
				if (!pub2.Publish(msg))
				  break;
				std::cout << "Publishing hello on topic [" << topic2 << "]" << std::endl;
				std::this_thread::sleep_for(std::chrono::milliseconds(10));
			}
			if(det->id == BID3){
				vec.set_x(cx);
				vec.set_y(cy);
				vec.set_z(0);
				quat.set_x(0);
				quat.set_y(0);
				quat.set_z(angle);
				quat.set_w(0);
				if (!pub3.Publish(msg))
				  break;
				std::cout << "Publishing hello on topic [" << topic3 << "]" << std::endl;
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
			}
        }
        zarray_destroy(detections);

        imshow("Tag Detections", frame);
        if (waitKey(1) >= 0)
            break;
    }

    apriltag_detector_destroy(td);
    if (!strcmp(famname, "tag36h11"))
        tag36h11_destroy(tf);
    else if (!strcmp(famname, "tag36h10"))
        tag36h10_destroy(tf);
    else if (!strcmp(famname, "tag36artoolkit"))
        tag36artoolkit_destroy(tf);
    else if (!strcmp(famname, "tag25h9"))
        tag25h9_destroy(tf);
    else if (!strcmp(famname, "tag25h7"))
        tag25h7_destroy(tf);
    getopt_destroy(getopt);

    return 0;
}
