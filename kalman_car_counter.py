from copy import copy

__author__ = 'Anthony'
import numpy as np
import cv2
import cv
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial.distance import pdist, squareform
from hungarian import linear_assignment

show_sub_img = False
show_raw_img = False
show_cluster_img = True
show_kalman_img = True
sub_window = "No background"

cap = cv2.VideoCapture("overpass.mp4")

fourcc = cv2.cv.CV_FOURCC('P', 'I', 'M', '1')
diff_out = cv2.VideoWriter("overpass_diff.avi", fourcc, 30, (1920, 1080), isColor=False)
cluster_out = cv2.VideoWriter("overpass_cluster.avi", fourcc, 30, (1920, 1080))
kalman_out = cv2.VideoWriter("overpass_kalman.avi", fourcc, 30, (1920, 1080))


if show_cluster_img:
    cluster_window = "Clusters"
def frame_diff(old, new):
        diff_frame = cv2.absdiff(cv2.cvtColor(old, cv2.COLOR_BGR2GRAY), cv2.cvtColor(new, cv2.COLOR_BGR2GRAY))
        #kernel = np.ones((3,3),np.uint8)
        #new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
        #blur_frame = new #cv2.morphologyEx(new, cv2.MORPH_OPEN, kernel)
        if show_sub_img:
            cv2.imshow(sub_window, diff_frame)
        diff_out.write(diff_frame)
        return diff_frame

def make_2d_kalman(x, y):
    kalman = cv.CreateKalman(4, 2, 0)
    kalman_state = cv.CreateMat(4, 1, cv.CV_32FC1)
    kalman_process_noise = cv.CreateMat(4, 1, cv.CV_32FC1)

    kalman_measurement = cv.CreateMat(2, 1, cv.CV_32FC1)

    # set previous state for prediction
    kalman.state_pre[0, 0] = x
    kalman.state_pre[1, 0] = y
    kalman.state_pre[2, 0] = 0
    kalman.state_pre[3, 0] = 0

    # set kalman transition matrix
    kalman.transition_matrix[0, 0] = 1
    kalman.transition_matrix[0, 1] = 0
    kalman.transition_matrix[0, 2] = .5
    kalman.transition_matrix[0, 3] = 0
    kalman.transition_matrix[1, 0] = 0
    kalman.transition_matrix[1, 1] = 1
    kalman.transition_matrix[1, 2] = 0
    kalman.transition_matrix[1, 3] = .5
    kalman.transition_matrix[2, 0] = 0
    kalman.transition_matrix[2, 1] = 0
    kalman.transition_matrix[2, 2] = 0
    kalman.transition_matrix[2, 3] = 1
    kalman.transition_matrix[3, 0] = 0
    kalman.transition_matrix[3, 1] = 0
    kalman.transition_matrix[3, 2] = 0
    kalman.transition_matrix[3, 3] = 1

    # set Kalman Filter
    cv.SetIdentity(kalman.measurement_matrix, cv.RealScalar(1))
    cv.SetIdentity(kalman.process_noise_cov, cv.RealScalar(.01))
    cv.SetIdentity(kalman.measurement_noise_cov, cv.RealScalar(.01))
    cv.SetIdentity(kalman.error_cov_post, cv.RealScalar(1))
    return kalman, kalman_measurement, kalman_state, kalman_process_noise


#
# params for ShiTomasi corner detection
feature_params = dict(maxCorners=500,
                      qualityLevel=.5,
                      minDistance=10,)
                      #blockSize=7)

# params for subpix corner refinement.
subpix_params = dict(zeroZone=(-1,-1),winSize=(10,10),
                     criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,20,0.03))

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))



# Create some random colors
color = np.random.randint(0, 255, (100, 3))

#
count = 0
tracks = []
features = []
kalmans = []
ret, raw_frame = cap.read()
ret, raw_frame2 = cap.read()
frame = frame_diff(raw_frame, raw_frame2)
while True:
    cv2.imshow("raw_video", raw_frame)
    ret, raw_frame2 = cap.read()
    old_frame = frame
    frame = frame_diff(raw_frame, raw_frame2)
    raw_frame = raw_frame2
    if True:#features is None or len(features) <= 2:
        features = cv2.goodFeaturesToTrack(frame, **feature_params)
        if features is None:
            continue
        if features is not None and len(features) > 3:
            cv2.cornerSubPix(frame, features, **subpix_params)
            tracks = [[p] for p in features.reshape((-1,2))]  # reshape features into pairs.
    #else:
        tmp = np.float32(features).reshape(-1, 1, 2)
        # calculate optical flow
        new_features, lk_status, lk_error = cv2.calcOpticalFlowPyrLK(old_frame,
                                                                 frame,
                                                                 tmp,
                                                                 None,
                                                                 **lk_params)

        # remove points that are "lost"
        features = [point[0] for (status, point) in zip(lk_status, new_features) if status]

        new_features = np.array(new_features).reshape((-1, 2))

        if not len(features) > 2:
            continue
        cluster_assignments = fclusterdata(features, 50, criterion='distance')
        if show_cluster_img:
            cluster_frame = copy(raw_frame2)
            for assignment, feature in zip(cluster_assignments, features):
                if assignment < len(color):
                    cv2.circle(cluster_frame, (int(feature[0]),
                                           int(feature[1])), 5, color[assignment], 10)
            #cv2.imshow('Clusters', cluster_frame)
            cluster_out.write(cluster_frame)

    clusters = []
    for i in range(max(cluster_assignments)):
        clusters.append([])

    for assignment, data in zip(cluster_assignments, features):
        clusters[assignment-1].append(data)


    large_clusters = [cluster for cluster in clusters if len(cluster) > 1]
    cluster_means = []
    for cluster in large_clusters:
        mean = np.mean(cluster, axis=0)
        cluster_means.append(mean)

    if not kalmans:  # if we aren't tracking any cars, see if there are any cars to track
        kalmans = [make_2d_kalman(point[0], point[1]) for point in cluster_means]
        lost = [0] * len(kalmans)


    # kalman predict
    predictions = [cv.KalmanPredict(kalman[0]) for kalman in kalmans]
    estimates = [(prediction[0, 0], prediction[1, 0]) for prediction in predictions]

    # perform linear assignment
    if estimates:
        dist = pdist(cluster_means + estimates)
        points_found = len(cluster_means)
        #dist = pdist([[1,1], [1.2,1.2], [3,3], [25,25], [24,26],[1.25,1.25], [1.3,1.3]])
        square_dist = squareform(dist)
        chopped = square_dist[:points_found, points_found:] #
        assignments = linear_assignment(chopped)  # we now have a list of pairs for each point.
        #print assignments
        new = range(points_found)
        successfully_tracked = []
        for assignment in assignments:
            new.remove(assignment[0])
            if square_dist[assignment[0], assignment[1]] < 50:
                successfully_tracked.append(assignment)
            else:
                lost[assignment[1]] += 1
    else:
        assignments = np.ndarray([])
    if assignments.size == 0:
        lost = [l+1 for l in lost]
    #next loops estimates
    # kalman measurement updates
    states = []
    for assignment in successfully_tracked:  # measurement update
        x = cluster_means[assignment[0]][0]
        y = cluster_means[assignment[0]][1]
        assigned_kalman = kalmans[assignment[1]]
        assigned_kalman[1][0, 0] = x
        assigned_kalman[1][1, 0] = y
        corrected = cv.KalmanCorrect(assigned_kalman[0], assigned_kalman[1])
        states.append((corrected[0, 0], corrected[1, 0]))
        lost[assignment[1]] = 0

    if estimates:
        for new_point in new:
            new_filter = make_2d_kalman(*cluster_means[new_point])
            prediction = cv.KalmanPredict(kalman[0])
            estimates.append((prediction[0, 0], prediction[1, 0]))
            kalmans.append(new_filter)
            lost.append(0)

    remove_idxs = []
    #print lost
    for idx, lost_count in enumerate(lost):
        if lost_count > 6:
            remove_idxs.append(idx)
    for idx in remove_idxs[::-1]:
        lost.pop(idx)
        kalmans.pop(idx)

    kal_idx = 0
    #print estimates
    if show_kalman_img:
        kalman_img = copy(raw_frame2)
        for point in estimates:
            point = int(point[0]), int(point[1])
            cv2.circle(kalman_img, point, 6, (255, 0, 0),3)

        #cv2.imshow("Kalman Centers", kalman_img)
        kalman_out.write(kalman_img)



    k = cv2.waitKey(30)
    if k == 27:
        break

diff_out.release()
cluster_out.release()
kalman_out.release()
print "goodbye"

