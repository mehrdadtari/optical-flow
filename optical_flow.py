import cv2
import numpy as np

def onLeftClick(event, x, y, flag, param):
    global ix, iy, k
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
        k = False

ix, iy, k = 200, 200, True
cv2.namedWindow('live_stream', cv2.WINDOW_NORMAL)
cv2.namedWindow("optical_flow_wind", cv2.WINDOW_NORMAL)
cv2.setMouseCallback('live_stream', onLeftClick)

strm = cv2.VideoCapture(0)

while True:
    _, frm = strm.read()
    cv2.imshow("live_stream", frm)
    
    if cv2.waitKey(1) == 27 or k == False:
        prev_img = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        break

prev_pts = np.array([[ix, iy]], dtype="float32").reshape(-1,1,2)

mask = np.zeros_like(frm)
        
while True:
    _, frm2 = strm.read()
    new_img = cv2.cvtColor(frm2, cv2.COLOR_BGR2GRAY)
    new_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_img,
                                                   new_img,
                                                   prev_pts,
                                                   None,
                                                   maxLevel=1,
                                                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.08))
    
    cv2.circle(mask, (int(new_pts.ravel()[0]), int(new_pts.ravel()[1])), 2, (0,255,0), 2)
    
    motion_tracked = cv2.addWeighted(frm2, 0.7, mask, 0.3, 0.1)
    
    cv2.imshow("optical_flow_wind", mask)
    cv2.imshow("live_stream", motion_tracked)
    
    prev_img = new_img.copy()
    prev_pts = new_pts.copy()
    
    if cv2.waitKey(1) == 27:
        strm.release()
        cv2.destroyAllWindows()
        break