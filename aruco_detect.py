import cv2

cap = cv2.VideoCapture(0)

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

parameters = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(
    dictionary,
    parameters
)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(
            frame,
            corners,
            ids
        )

        for marker_id, marker_corners in zip(ids, corners):
            print("Detected ID:", marker_id[0])
            print("Corners:")
            print(marker_corners)

    cv2.imshow("ArUco Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
