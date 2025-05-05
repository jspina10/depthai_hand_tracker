import cv2
import depthai as dai
import numpy as np

pipeline = dai.Pipeline()

# Define sources and outputs
rgbRight = pipeline.create(dai.node.ColorCamera)
rgbLeft = pipeline.create(dai.node.ColorCamera)
xRightOut = pipeline.create(dai.node.XLinkOut)
xLeftOut = pipeline.create(dai.node.XLinkOut)

xRightOut.setStreamName("rgb_right")
xLeftOut.setStreamName("rgb_left")

rgbRight.setBoardSocket(dai.CameraBoardSocket.CAM_B)
rgbLeft.setBoardSocket(dai.CameraBoardSocket.CAM_C)

rgbRight.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
rgbLeft.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
rgbRight.setPreviewSize(700, 700)

rgbRight.preview.link(xRightOut.input)
rgbLeft.preview.link(xLeftOut.input)


device_infos = dai.Device.getAllAvailableDevices()
if len(device_infos) == 0:
    raise RuntimeError("No devices found!")
else:
    print("Found", len(device_infos), "devices")
    for device_info in device_infos:
        print("device info")
        print(device_info)

with dai.Device(pipeline) as device:

    print('USB speed:',device.getUsbSpeed())
    # Output queue will be used to get the disparity frames from the outputs defined above
    q_right = device.getOutputQueue(name="rgb_right", maxSize=4, blocking=False)
    q_left = device.getOutputQueue(name="rgb_left", maxSize=4, blocking=False)

    while True:
        in_right = q_right.get()
        in_left = q_left.get()
        frame_right = in_right.getCvFrame()
        frame_left = in_left.getCvFrame()
        
        cv2.imshow("rgb_right", frame_right)
        cv2.imshow("rgb_left", frame_left)

        if cv2.waitKey(1) == ord('q'):
            break