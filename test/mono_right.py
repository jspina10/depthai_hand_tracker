import cv2
import depthai as dai
import numpy as np

pipeline = dai.Pipeline()

# Define sources and outputs
monoRight = pipeline.create(dai.node.MonoCamera)
xout = pipeline.create(dai.node.XLinkOut)

xout.setStreamName("mono_right")

monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setCamera("right")

monoRight.out.link(xout.input)


device_infos = dai.Device.getAllAvailableDevices()
if len(device_infos) == 0:
    raise RuntimeError("No devices found!")
else:
    print("Found", len(device_infos), "devices")
    for device_info in device_infos:
        print("device info")
        print(device_info)

with dai.Device(pipeline) as device:
    print(f"input queues: {device.getInputQueueNames()}")
    print(f"output queues: {device.getOutputQueueNames()}")

    # Output queue will be used to get the disparity frames from the outputs defined above
    q = device.getOutputQueue(name="mono_right", maxSize=4, blocking=False)

    while True:
        in_right = q.get()
        frame = in_right.getFrame()

        cv2.imshow("mono_right", frame)

        if cv2.waitKey(1) == ord('q'):
            break