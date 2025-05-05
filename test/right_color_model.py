import cv2
import depthai as dai
import numpy as np
import blobconverter

labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

pipeline = dai.Pipeline()

# Define sources and outputs
rgbRight = pipeline.create(dai.node.ColorCamera)
rgbRight.setBoardSocket(dai.CameraBoardSocket.CAM_B)
rgbRight.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
rgbRight.setPreviewSize(300, 300)
rgbRight.setInterleaved(False)

detector = pipeline.create(dai.node.MobileNetDetectionNetwork)
detector.setConfidenceThreshold(0.5)
detector.setBlobPath(blobconverter.from_zoo(name="mobilenet-ssd", shaves=6))
rgbRight.preview.link(detector.input)

xRightOut = pipeline.create(dai.node.XLinkOut)
xRightOut.setStreamName("rgb_right")
detector.passthrough.link(xRightOut.input)

xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")
detector.out.link(xout_nn.input)

with dai.Device(pipeline) as device:

    print('USB speed:',device.getUsbSpeed())
    # Output queue will be used to get the disparity frames from the outputs defined above
    q_right = device.getOutputQueue(name="rgb_right", maxSize=4, blocking=False)
    nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    while True:
        if nn.has():
            dets = nn.get().detections
            in_right = q_right.get()
            frame_right = in_right.getCvFrame()

            for detection in dets:
                ymin = int(300*detection.ymin)
                xmin = int(300*detection.xmin)
                cv2.putText(frame_right, labelMap[detection.label], (xmin + 10, ymin + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
                cv2.putText(frame_right, f"{int(detection.confidence * 100)}%", (xmin + 10, ymin + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
                cv2.rectangle(frame_right, (xmin, ymin), (int(300*detection.xmax), int(300*detection.ymax)), (255,255,255), 2)
        
            cv2.imshow("rgb_right", frame_right)

        if cv2.waitKey(1) == ord('q'):
            break