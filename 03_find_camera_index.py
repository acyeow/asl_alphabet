import cv2 as cv

def find_camera_id():
    for camera_id in range(10):  # Try IDs from 0 to 9
        cap = cv.VideoCapture(camera_id)
        if cap.isOpened():
            print(f"Camera found at ID {camera_id}")
            cap.release()
            return camera_id
        cap.release()
    print("No camera found")
    return None

camera_id = find_camera_id()
if camera_id is not None:
    cap = cv.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow('frame', gray)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
else:
    print("No camera available")