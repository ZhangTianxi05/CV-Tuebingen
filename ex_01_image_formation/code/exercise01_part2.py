import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from IPython.display import HTML
from matplotlib import animation
from matplotlib.patches import Polygon
import cv2

# 这里一定要加入plt.show(block=True),不然会直接退出
# Load images
img1 = cv2.cvtColor(cv2.imread("./image1.png"), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread("./image2.png"), cv2.COLOR_BGR2RGB)
# Load matching points
npz_file = np.load("./panorama_points.npz")
points_source = npz_file["points_source"]
points_target = npz_file["points_target"]
# Let's visualize the images
f = plt.figure(figsize=(15, 5))
f.show()
ax1 = f.add_subplot(121)
ax2 = f.add_subplot(122)
ax1.imshow(img1)
ax2.imshow(img2)


###########################
##### Helper Function #####
###########################
def draw_matches(img1, points_source, img2, points_target):
    """Returns an image with matches drawn onto the images."""
    r, c = img1.shape[:2]
    r1, c1 = img2.shape[:2]

    output_img = np.zeros((max([r, r1]), c + c1, 3), dtype="uint8")
    output_img[:r, :c, :] = np.dstack([img1])
    output_img[:r1, c : c + c1, :] = np.dstack([img2])

    for p1, p2 in zip(points_source, points_target):
        (x1, y1) = p1[:2]
        (x2, y2) = p2[:2]

        cv2.circle(output_img, (int(x1), int(y1)), 10, (0, 255, 255), 10)
        cv2.circle(output_img, (int(x2) + c, int(y2)), 10, (0, 255, 255), 10)

        cv2.line(
            output_img, (int(x1), int(y1)), (int(x2) + c, int(y2)), (0, 255, 255), 5
        )

    return output_img


f = plt.figure(figsize=(20, 10))
vis = draw_matches(img1, points_source[:5], img2, points_target[:5])
plt.imshow(vis)
plt.show(block=True)


###########################
#### Exercise Function ####
###########################
def get_Ai(xi_vector, xi_prime_vector):
    """Returns the A_i matrix discussed in the lecture for input vectors.

    Args:
        xi_vector (array): the x_i vector in homogeneous coordinates
        xi_vector_prime (array): the x_i_prime vector in homogeneous coordinates
    """
    assert xi_vector.shape == (3,) and xi_prime_vector.shape == (3,)
    zero_vector = np.zeros((3,), dtype=np.float32)
    xi, yi, wi = xi_prime_vector

    Ai = np.stack(
        [
            np.concatenate([zero_vector, -wi * xi_vector, yi * xi_vector]),
            np.concatenate([wi * xi_vector, zero_vector, -xi * xi_vector]),
            # np.concatenate([-yi*xi_vector, xi*xi_vector, zero_vector]) this is not needed, so we comment it out
        ]
    )
    assert Ai.shape == (2, 9)
    return Ai


###########################
#### Exercise Function ####
###########################
def get_A(points_source, points_target):
    """Returns the A matrix discussed in the lecture.

    Args:
        points_source (array): 3D homogeneous points from source image
        points_target (array): 3D homogeneous points from target image
    """
    N = points_source.shape[0]
    correspondence_pairs = zip(points_source, points_target)

    A = np.concatenate([get_Ai(p1, p2) for (p1, p2) in correspondence_pairs])
    assert A.shape == (2 * N, 9)
    return A
###########################
#### Exercise Function ####
###########################
def get_homography(points_source, points_target):
    ''' Returns the homography H.
    
    Args:
        points_source (array): 3D homogeneous points from source image
        points_target (array): 3D homogeneous points from target image        
    '''
    
    # Insert your code here
    A = get_A(points_source, points_target)
    vh = np.linalg.svd(A)[2]
    H = vh[-1].reshape(3, 3)
    return H
###########################
##### Helper Function #####
###########################
def stich_images(img1, img2, H):
    ''' Stitches together the images via given homography H.

    Args:
        img1 (array): image 1
        img2 (array): image 2
        H (array): homography
    '''

    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0,0], [0, rows1],[cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)

    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1,list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min,-y_min]

    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1

    return output_img
H = get_homography(points_target, points_source)
stiched_image = stich_images(img1, img2, H)
fig = plt.figure(figsize=(15, 10))
fig.suptitle("Stiched Panorama")
plt.imshow(stiched_image)

###########################
#### Exercise Function ####
###########################
# Load images
img1 = cv2.cvtColor(cv2.imread('./image1.png'), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread('./image2.png'), cv2.COLOR_BGR2RGB)
# Let's visualize the images
f = plt.figure(figsize=(15, 5))
ax1 = f.add_subplot(121)
ax2 = f.add_subplot(122)
ax1.imshow(img1)
ax2.imshow(img2)
plt.show(block=True)
###########################
##### Helper Function #####
###########################
def get_keypoints(img1, img2):
    orb = cv2.ORB_create(nfeatures=2000)

    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)

    # Find matching points
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)
    p_source = np.float32([ keypoints1[good_match.queryIdx].pt for good_match in good ]).reshape(-1,2)
    p_target = np.float32([ keypoints2[good_match.trainIdx].pt for good_match in good ]).reshape(-1,2)
    N = p_source.shape[0]
    p_source = np.concatenate([p_source, np.ones((N, 1))], axis=-1)
    p_target = np.concatenate([p_target, np.ones((N, 1))], axis=-1)
    return p_source, p_target
p_source, p_target = get_keypoints(img1, img2)
H = get_homography(p_target, p_source)
stiched_image = stich_images(img1, img2, H)
fig = plt.figure(figsize=(15, 10))
fig.suptitle("Stiched Panorama")
plt.imshow(stiched_image)