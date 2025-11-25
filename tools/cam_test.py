import cv2
import time

print("Starting webcam probe (indices 0-3)...")
found = False
for idx in range(4):
    print(f"Trying camera index {idx}...")
    cap = cv2.VideoCapture(idx)
    # small delay to allow device to initialize
    time.sleep(0.5)
    if not cap.isOpened():
        print(f"  index {idx}: not opened")
        cap.release()
        continue
    # try to read a frame
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"  index {idx}: opened but failed to read frame")
        cap.release()
        continue
    h, w = frame.shape[:2]
    print(f"  index {idx}: success — frame size {w}x{h}")
    fname = f"cam_test_{idx}.jpg"
    cv2.imwrite(fname, frame)
    print(f"  saved test frame to {fname}")
    found = True
    cap.release()
    break

if not found:
    print("No working camera found at indices 0-3.")

print("Probe complete.")
