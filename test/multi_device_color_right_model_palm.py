#!/usr/bin/env python3

import depthai as dai
import threading
import contextlib
import cv2
import time
import blobconverter

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
# PALM_DETECTION_MODEL = str(SCRIPT_DIR / "models/palm_detection_sh4.blob")
PALM_DETECTION_MODEL = "../models/palm_detection_sh4.blob"
print(f"aaaaaaaaaaaaa {PALM_DETECTION_MODEL}")


# This can be customized to pass multiple parameters
def getPipeline():
    pipeline = dai.Pipeline()

    # Define sources and outputs
    rgbRight = pipeline.create(dai.node.ColorCamera)
    rgbRight.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    rgbRight.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
    rgbRight.setPreviewSize(128, 128)
    rgbRight.setInterleaved(False)

    # detector = pipeline.create(dai.node.MobileNetDetectionNetwork)
    detector = pipeline.createNeuralNetwork()
    # detector.setConfidenceThreshold(0.5)
    # detector.setBlobPath(blobconverter.from_zoo(name="mobilenet-ssd", shaves=6))
    detector.setBlobPath(PALM_DETECTION_MODEL)
    rgbRight.preview.link(detector.input)

    xRightOut = pipeline.create(dai.node.XLinkOut)
    xRightOut.setStreamName("rgb_right")
    detector.passthrough.link(xRightOut.input)

    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("nn")
    detector.out.link(xout_nn.input)

    return pipeline


def worker(device_info, stack, devices):
    openvino_version = dai.OpenVINO.Version.VERSION_2021_4
    usb2_mode = False
    device = stack.enter_context(dai.Device(openvino_version, device_info, usb2_mode))

    # Note: currently on POE, DeviceInfo.getMxId() and Device.getMxId() are different!
    print("=== Connected to " + device_info.getMxId())
    print('USB speed:', device.getUsbSpeed())
    device.startPipeline(getPipeline())

    # Output queue will be used to get the rgb frames from the output defined above
    devices[device.getMxId()] = {
        'rgb': device.getOutputQueue(name="rgb_right"),
        'nn': device.getOutputQueue(name="nn"),
    }


# https://docs.python.org/3/library/contextlib.html#contextlib.ExitStack
with contextlib.ExitStack() as stack:
    device_infos = dai.Device.getAllAvailableDevices()
    if len(device_infos) == 0:
        raise RuntimeError("No devices found!")
    else:
        print("Found", len(device_infos), "devices")
    devices = {}
    threads = []

    for device_info in device_infos:
        thread = threading.Thread(target=worker, args=(device_info, stack, devices))
        thread.start()
        threads.append(thread)

    for t in threads:
        t.join() # Wait for all threads to finish (to connect to devices)

    while True:
        for mxid, q in devices.items():
            if q['nn'].has():
                dets = q['nn'].get().detections
                frame = q['rgb'].get().getCvFrame()

                for detection in dets:
                    ymin = int(300*detection.ymin)
                    xmin = int(300*detection.xmin)
                    # cv2.putText(frame, labelMap[detection.label], (xmin + 10, ymin + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
                    cv2.putText(frame, f"{int(detection.confidence * 100)}%", (xmin + 10, ymin + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
                    cv2.rectangle(frame, (xmin, ymin), (int(300*detection.xmax), int(300*detection.ymax)), (255,255,255), 2)
                # Show the frame

                cv2.imshow(f"Preview - {mxid}", frame)

        if cv2.waitKey(1) == ord('q'):
            break
