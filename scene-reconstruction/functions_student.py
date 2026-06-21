import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm

try:
    import gco
except ImportError:
    pass # handle gco missing gracefully if needed

def show_matching_result(img1, img2, keypoints1, keypoints2):
    """
    Plot the images and their corresponding matching points.
    
    Args:
        img1 (np.ndarray): First input image.
        img2 (np.ndarray): Second input image.
        keypoints1 (np.ndarray): Array of matching keypoints from img1.
        keypoints2 (np.ndarray): Array of matching keypoints from img2.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) if img1.ndim == 3 else img1, cmap='gray')
    axes[0].scatter(keypoints1[:, 0], keypoints1[:, 1], c='red', s=10)
    axes[0].set_title('Image 1 - matched keypoints')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) if img2.ndim == 3 else img2, cmap='gray')
    axes[1].scatter(keypoints2[:, 0], keypoints2[:, 1], c='red', s=10)
    axes[1].set_title('Image 2 - matched keypoints')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    # side-by-side with lines
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas_height = max(h1, h2)
    canvas_width = w1 + w2
    if img1.ndim == 3:
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=img1.dtype)
        canvas[:h1, :w1] = img1
        canvas[:h2, w1:] = img2
    else:
        canvas = np.zeros((canvas_height, canvas_width), dtype=img1.dtype)
        canvas[:h1, :w1] = img1
        canvas[:h2, w1:] = img2

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB) if canvas.ndim == 3 else canvas, cmap='gray')
    for pt1, pt2 in zip(keypoints1, keypoints2):
        ax.plot([pt1[0], pt2[0] + w1], [pt1[1], pt2[1]], 'g-', linewidth=0.5, alpha=0.5)
    ax.scatter(keypoints1[:, 0], keypoints1[:, 1], c='red', s=10)
    ax.scatter(keypoints2[:, 0] + w1, keypoints2[:, 1], c='blue', s=10)
    ax.set_title('Matches')
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def get_affine_transformation(
    points_in: NDArray[np.float32],
    points_out: NDArray[np.float32],
):
    """
    Computes an affine transformation from points_in to points_out using least squares.
    
    Args:
        points_in (NDArray[np.float32]): Array of input coordinates.
        points_out (NDArray[np.float32]): Array of target coordinates.
        
    Returns:
        NDArray: Computed affine transformation matrix.
    """
    # transform to homogenous coordinates
    points_in = np.hstack((points_in, np.ones((len(points_in), 1))), dtype=np.float32)
    points_out = np.hstack(
        (points_out, np.ones((len(points_out), 1))),
        dtype=np.float32,
    )

    # solve the least-squares problem A.T@Ax = A.Tb
    resulting = np.linalg.solve(points_in.T @ points_in, points_in.T @ points_out)

    return np.round(resulting.T, decimals=5)


def transform_points(points, matrix):
    """
    Transforms points using a given 3x3 transformation matrix.
    
    Args:
        points (NDArray): Points to transform.
        matrix (NDArray): 3x3 transformation matrix.
        
    Returns:
        NDArray: Transformed coordinates.
    """
    # homogneous coordinates
    points = np.hstack((points, np.ones((len(points), 1))), dtype=np.float32)

    # transform the points
    points = (matrix @ points.T).T
    points = points[:, :2] / points[:, 2, np.newaxis]
    return points


def get_keypoints(image, filtering=False, sigma=3):
    """
    Returns the keypoints of the image using the SIFT algorithm.
    
    Args:
        image (np.ndarray): Input image.
        filtering (bool): Optional flag to apply Gaussian filtering.
        sigma (float): Sigma for Gaussian filtering if enabled.
        
    Returns:
        tuple: (keypoints array, descriptors array)
    """
    # apply an optional smoothing to reduce the amount of found keypoints
    if filtering:
        image = gaussian_filter(image.astype(np.float32), sigma=sigma)
        image = image.astype(np.uint8)

    # detect keypoints and compute descriptors with SIFT
    sift = cv2.ORB_create(nfeatures=5000, fastThreshold=5, edgeThreshold=15, patchSize=31)
    keypoints, descriptors = sift.detectAndCompute(image, None)

    # transform keypoints to numpy array
    keypoints_array = np.array([kp.pt for kp in keypoints], dtype=np.float32)

    # return the keypoints and the descriptors
    return keypoints_array, descriptors


def intersect2d(array1, array2):
    """
    Calculates the intersection over the rows of two arrays.
    
    Args:
        array1 (NDArray): First array.
        array2 (NDArray): Second array.
        
    Returns:
        NDArray: Array containing intersecting rows.
    """
    # tests which entries are the same
    test = array1[:, None] == array2  # needs [:, None] to induce broadcasting

    # selects only the rows of array2 that appear in array1
    matches = np.all(test, axis=2)
    rows_in_array2 = np.any(matches, axis=0)
    return array2[rows_in_array2]


def matching(descriptors_1, descriptors_2, max_ratio=0.7, cross_checking=True):
    """
    Matches the descriptors against each other.
    Returns the best match for each descriptor, if it is significant.
    The significance is defined by the max_ratio: distance_1 / distance_2 < max_ratio.
    Optional cross-checking of matches.
    
    Args:
        descriptors_1 (NDArray): Descriptors from the first image.
        descriptors_2 (NDArray): Descriptors from the second image.
        max_ratio (float): Maximum ratio between nearest neighbor distances for significance testing.
        cross_checking (bool): If True, validates matches bidirectionally.
        
    Returns:
        NDArray: Array of match indices for descriptors.
    """
    # choose distance metric: Hamming for binary descriptors (ORB/AKAZE), Euclidean for SIFT
    if descriptors_1.dtype == np.uint8 and descriptors_2.dtype == np.uint8:
        metric = 'hamming'
    else:
        metric = 'euclidean'

    # calculate the distance between all pairs
    distances = cdist(descriptors_1, descriptors_2, metric=metric)

    # get the first and second match (only sorts up to the kth smallest entry, in O(n + k log(k)))
    nearest = np.partition(distances, 1, axis=1)[:, :2]

    # get the ratios
    ratios = nearest[:, 0] / (nearest[:, 1] + 1e-10)

    # cutoff at max_ratio
    valid = ratios < max_ratio
    matched_indices_2 = np.argmin(distances, axis=1)

    # produce the final matches
    final_matches = np.column_stack((np.arange(len(descriptors_1)), matched_indices_2))
    final_matches = final_matches[valid]

    # cross_checking
    if cross_checking:
        nearest_rev = np.partition(distances.T, 1, axis=1)[:, :2]
        ratios_rev = nearest_rev[:, 0] / (nearest_rev[:, 1] + 1e-10)
        valid_rev = ratios_rev < max_ratio
        matched_indices_1 = np.argmin(distances.T, axis=1)
        final_matches2 = np.column_stack((matched_indices_1, np.arange(len(descriptors_2))))
        final_matches2 = final_matches2[valid_rev]

        # return the intersection of the two matches arrays
        final_matches = intersect2d(final_matches, final_matches2)

    return final_matches





def compute_fundamental_matrix(points1, points2):
    """Compute the fundamental matrix given the point correspondences.

    y'.T @ F @ y = 0

    y' = (y1', y2', 1)
    y = (y1, y2, 1)
    
    Parameters
    ------------
    points1, points2 - array with shape [n, 3]
        corresponding points in images represented as 
        homogeneous coordinates
    """
    # TODO: Implement the algebraic SVD trick to find the fundamental matrix
    rows, cols = points1.shape

    if rows != points2.shape[0]:
        raise ValueError("points1 and points2 should have the same shape!")

    #create the design matrix:
    A = np.zeros((rows, 9))

    #design each row of matrix A through homogeneous coordinates multiplication between points1 and points2
    for i in range(rows):
        p1 = points1[i]
        p2 = points2[i]

        x, y, w = p1[0], p1[1], p1[2]
        xp, yp, wp = p2[0], p2[1], p2[2]

        row = [xp * x, xp * y, xp * w,
            yp * x, yp * y, yp * w,
            wp * x, wp * y, wp * w]

        A[i] = row

    #apply SVD to A to find the solution vector which is created by the smallest singular value (Solve homogeneous system Af=0 via SVD)
    U, S, Vt = np.linalg.svd(A)

    #extract null space solution (right singular vector of smallest singular value)
    f = Vt[-1]

    #reshape vector into 3x3 matrix F
    F = f.reshape(3,3)

    #enforce rank-2 constraint via SVD of F by: performing an SVD of F
    UF, SF, VtF = np.linalg.svd(F)

    #setting the smallest singular value to 0,
    SF[-1] = 0

    #rebuilding F with rank = 2
    F = UF @ np.diag(SF) @ VtF

    return F
    
    
def compute_fundamental_matrix_normalized(points1, points2):
    """
    Normalize points by calculating the centroid, subtracting
    it from the points and scaling the points such that the distance
    from the origin is sqrt(2)

    Parameters
    ------------
    points1, points2 - with shape [n, 2]
    """
    # TODO: Normalize the points, compute the fundamental matrix using compute_fundamental_matrix, and denormalize it
    def _normalize(pts):
        pts = pts.astype(np.float64)
        centroid = np.mean(pts, axis=0)
        shifted = pts - centroid
        mean_dist = np.mean(np.sqrt(np.sum(shifted ** 2, axis=1)))
        scale = np.sqrt(2) / mean_dist if mean_dist > 1e-10 else 1.0
        T = np.array([
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1]
        ])
        pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
        pts_norm = (T @ pts_h.T).T
        return pts_norm, T

    points1_norm, T1 = _normalize(points1)
    points2_norm, T2 = _normalize(points2)

    F_norm = compute_fundamental_matrix(points1_norm, points2_norm)

    # denormalize: F = T2^T @ F_norm @ T1
    F = T2.T @ F_norm @ T1

    # normalize scale so that F[2,2] is 1 (common convention)
    if abs(F[2, 2]) > 1e-10:
        F = F / F[2, 2]

    return F


# ==============================================================================
# EXERCISE 2: FUNCTIONS SUMMARY NOTES (by Ammara Ansari)
# ==============================================================================
#
# 1. get_max_expected_disparity
#    - Analyzes horizontal pixel shifts between matching keypoints across both images.
#    - Uses the 99th percentile to establish a robust maximum search range while filtering out tracking outliers.
#
# 2. compute_disparity_map
#    - Performs dense block matching by sliding a local window along corresponding image scanlines.
#    - Evaluates pixel similarity using the Sum of Absolute Differences (SAD) to select the optimal pixel shift.
#
# 3. cross_check_disparities
#    - Performs a bidirectional validation check to enforce Left-Right consistency between matching views.
#    - Invalidates mismatched pixels and occluded regions by resetting non-symmetric assignments to zero.
#
# 4. estimate_depth
#    - Converts geometric pixel disparities into absolute physical distance from the camera centers.
#    - Multiplies focal length by baseline distance and divides by valid disparity values to calculate metric depth.
#
# ==============================================================================

def get_max_expected_disparity(points1: np.ndarray, points2: np.ndarray) -> int:
    """
    Calculates the maximum expected horizontal disparity from matched points.

    It computes the absolute horizontal difference for each matched pair,
    takes the 99th percentile of these disparities.

    Args:
        points1 (np.ndarray): A (N, 2) NumPy array of (x, y) coordinates for
                              matched keypoints in the first image.
        points2 (np.ndarray): A (N, 2) NumPy array of (x, y) coordinates for
                              matched keypoints in the second image,
                              corresponding to pts1_xy.

    Returns:
        int: The estimated maximum expected disparity, rounded up to the nearest integer.
    """
    # TODO: Calculate maximum expected horizontal disparity from matched points using 99th percentile

    #extract horizontal coordinates
    x_coords_img1 = points1[:, 0]
    x_coords_img2 = points2[:, 0]

    #calculate the absolute horizontal differences
    pixel_shifts = np.abs(x_coords_img1 - x_coords_img2)
    
    #compute the 99th percentile - filter out extreme outliers and round up the scores to not have floating point numbers
    upper_bound_shift = np.percentile(pixel_shifts, 99)

    return int(np.ceil(upper_bound_shift))

def compute_disparity_map(img_fixed: np.ndarray, img_search: np.ndarray, window_size: int, max_disparity: int, direction: str = "L->R") -> np.ndarray:
    """
    Computes the disparity map using Sum of Absolute Differences (SAD) over a local window.
    Searches along the same scanline.
    
    Args:
        img_fixed (np.ndarray): The reference image against which blocks are matched.
        img_search (np.ndarray): The target image to search for matches.
        window_size (int): Dimensions of the block matching window (e.g. 5x5).
        max_disparity (int): The maximum disparity search range.
        direction (str): 'L->R' if img_fixed is the left image, 'R->L' if it is the right image.
        
    Returns:
        np.ndarray: Calculated disparity map of shape (H, W).
    """
    # TODO: Implement disparity map computation using Sum of Absolute Differences (SAD)

    #initialize the disparity map and calculate padding
    height, width = img_fixed.shape[:2]
    disp_output = np.zeros((height, width), dtype=np.float32)
    pad_offset = window_size // 2

    # Iterate over the image grid preserving boundary patch padding
    for r in range(pad_offset, height - pad_offset):
        for c in range(pad_offset, width - pad_offset):
            
            # Slice the baseline block window from the fixed view
            fixed_patch = img_fixed[r - pad_offset : r + pad_offset + 1, c - pad_offset : c + pad_offset + 1]

            minimum_sad = float('inf')
            optimal_disp = 0

            # Scan matching ranges up to the designated search ceiling
            for d_candidate in range(max_disparity + 1):
                # Calculate horizontal searching path depending on target view direction
                if direction == "L->R":
                    c_target = c - d_candidate
                else:
                    c_target = c + d_candidate
                
                # Verify that the target block sits completely within image frames
                if pad_offset <= c_target < width - pad_offset:
                    search_patch = img_search[r - pad_offset : r + pad_offset + 1, c_target - pad_offset : c_target + pad_offset + 1]

                    # Quantify differences via Sum of Absolute Differences
                    current_sad = np.sum(np.abs(fixed_patch - search_patch))
                    
                    # Track localized matching minimums
                    if current_sad < minimum_sad:
                        minimum_sad = current_sad
                        optimal_disp = d_candidate

            disp_output[r, c] = optimal_disp

    return disp_output

def cross_check_disparities(disparity_L: np.ndarray, disparity_R: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Performs cross-checking on L-R and R-L disparity maps to filter inconsistent matches.
    A match D_L(y, x) is consistent if D_R(y, x - D_L(y, x)) is close to D_L(y, x).

    Args:
        disparity_L (np.ndarray): Left-to-Right disparity map (H, W).
        disparity_R (np.ndarray): Right-to-Left disparity map (H, W).
        threshold (float): Maximum allowed absolute difference for consistency.

    Returns:
        np.ndarray: Cross-checked disparity map (H, W). Invalid matches are set to 0.
    """
    # TODO: Implement cross-checking logic
    #can be done in a vectorized way

    img_h, img_w = disparity_L.shape
    validated_disp = disparity_L.copy()
    
    #initialize a grid of column coordinates
    _, x_grid = np.indices(disparity_L.shape)

    #compute all target columns simultaneously, convert to integers for indexing
    projected_cols = np.round(x_grid - disparity_L).astype(int)
    
   # Generate look-up mask boundaries to capture structural overflows
    boundary_violations = (projected_cols < 0) | (projected_cols >= img_w)
    bounded_target_cols = np.clip(projected_cols, 0, img_w - 1)

    # Map corresponding Right-to-Left values back into spatial coordinates
    y_indices = np.arange(img_h)[:, None]
    mapped_R_disparities = disparity_R[y_indices, bounded_target_cols]

    # Validate structural consistency requirements against the deviation threshold
    mismatch_mask = (np.abs(disparity_L - mapped_R_disparities) > threshold) | boundary_violations

    # Nullify invalid/occluded coordinate sets to 0
    validated_disp[mismatch_mask] = 0.0

    return validated_disp


def estimate_depth(disparity_map: np.ndarray, focal_length_pixels: float, baseline_meters: float) -> np.ndarray:
    """
    Estimates scene depth from a disparity map.
    
    Args:
        disparity_map (np.ndarray): The computed disparity map.
        focal_length_pixels (float): Camera focal length in pixels (from camera calibration).
        baseline_meters (float): Distance between camera optical centers in meters.
        
    Returns:
        np.ndarray: A depth map representing Z coordinate values.
    """
    # TODO: Implement depth estimation from disparity

    #since previously we set invalid values to zero, now we have to handle division by zero
    depth_map = np.zeros_like(disparity_map, dtype=np.float32)

    #create a mask that selects values larger than zero
    active_pixels = disparity_map > 0

    #apply the stereo depth formula
    depth_map[active_pixels] = (focal_length_pixels * baseline_meters) / disparity_map[active_pixels]
    
    return depth_map


def apply_mrf(depth_map: np.ndarray, max_disparity: int,
              lambda_: int = 3) -> np.ndarray:
    """
    Apply a Markov Random Field to a depth map to fill holes and remove noise.
    Requires gco-wrapper.

    Args:
        depth_map (np.ndarray): The initial scene depth map.
        max_disparity (int): The maximum disparity label possible.
        lambda_ (int): Scaling smoothness penalty parameter.

    Returns:
        np.ndarray: The smoothed depth map after MRF application.
    """
    # TODO: Implement MRF using gco-wrapper
    h, w = depth_map.shape

    #scale to get rid of missing detail
    scale_factor = 10
    scaled_depth_map = np.round(depth_map * scale_factor).astype(np.int32)

    #find the valid range for the labels
    min_label = int(np.min(scaled_depth_map[scaled_depth_map > 0]))
    max_label = int(np.max(scaled_depth_map))

    # create an array using the prepared range
    depths = np.arange(min_label, max_label + 1)

    # compute unary and binary cost with the helper functions
    #we use the scaled depth map to find the unary cost, this way we avoid losing detail
    unary_costs = _calc_unary_costs(scaled_depth_map, depths)
    binary_costs = _calc_binary_costs(depths, lambda_)

    #set spatial weights to 1
    cost_v = np.ones((h - 1, w), dtype=np.int32)
    cost_h = np.ones((h, w - 1), dtype=np.int32)

    # create smoothed labels
    smoothed_labels = gco.cut_grid_graph(unary_costs, binary_costs, cost_v, cost_h)

    smoothed_labels = smoothed_labels.reshape(h, w)

    #rescale back to the original scale
    optimized_scaled_depths = depths[smoothed_labels]
    smoothed_depth_map = optimized_scaled_depths / scale_factor

    return smoothed_depth_map


def _calc_unary_costs(depth_map: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """
    Calculate the unary costs for the MRF.
    Costs are equal to the absolute difference between the current depth and the proposed depth.

    Args:
        depth_map (np.ndarray): The current depth map.
        depths (np.ndarray): Array of possible disparity/depth labels.

    Returns:
        np.ndarray: Calculated unary costs of shape (H, W, len(depths)).
    """
    # TODO: Calculate unary costs (absolute difference)
    # add a new axis to the depth_map to go from (H, W) to (H, W, 1), calculate the absolute value in a vectorized way
    unary_costs = np.abs(depth_map[:, :, np.newaxis] - depths)

    # where the depth_map had 0s, turn it back to 0
    holes = (depth_map == 0)[:, :, np.newaxis]
    unary_costs = np.where(holes, 0, unary_costs)

    # return as type int32 just to make sure it passes through gco-wrapper
    return unary_costs.astype(np.int32)


def _calc_binary_costs(depths: np.ndarray, lambda_: int) -> np.ndarray:
    """
    Calculate the binary costs for the MRF.
    Costs are equal to the absolute difference between the two depths, multiplied by lambda.

    Args:
        depths (np.ndarray): Array of possible depth/disparity labels.
        lambda_ (int): The smoothness penalty weight.

    Returns:
        np.ndarray: Pairwise cost matrix of shape (len(depths), len(depths)).
    """
    # TODO: Calculate binary costs (Potts model)
    # description says absolute difference between two depths multiplied by lambda, then the result is:
    binary_costs = np.abs(
        depths[:, np.newaxis] - depths[np.newaxis, :]) * lambda_

    # if we want a Potts model then uncomment this instead:
    # n = len(depths)
    # binary_costs = np.full((n, n), lambda_)
    # np.fill_diagonal(binary_costs, 0)
    return binary_costs.astype(np.int32)


def extract_surface_marching_squares(depth_map: np.ndarray,
                                     isovalue: float) -> list:
    """
    Implement the marching squares algorithm to extract isosurfaces from the depth map.

    Args:
        depth_map (np.ndarray): The smoothed depth map grid.
        isovalue (float): The threshold depth value corresponding to the surface.

    Returns:
        list: A list of line segments where each segment is a tuple of coordinates: [((x1, y1), (x2, y2)), ...]
    """
    # TODO: Implement the marching squares algorithm
    # Hint: You must resolve the ambiguous saddle cases (where opposite corners share the same sign)
    # by subsampling the center of the cell (i.e. computing the average of the 4 corners).

    h, w = depth_map.shape

    # define a lookup directory of states. 0 = top edge, 1 = right edge, 2 = bottom edge, 3 = left edge
    # states 5 and 10 are ambiguous saddle points and require center-point checking.
    states = {
        0: [],
        1: [[2, 3]],
        2: [[1, 2]],
        3: [[1, 3]],
        4: [[0, 1]],
        5: {
            "center_high": [[0, 3], [1, 2]],
            "center_low": [[0, 1], [2, 3]]
        },
        6: [[0, 2]],
        7: [[0, 3]],
        8: [[0, 3]],
        9: [[0, 2]],
        10: {
            "center_high": [[0, 1], [2, 3]],
            "center_low": [[0, 3], [1, 2]]
        },
        11: [[0, 1]],
        12: [[1, 3]],
        13: [[1, 2]],
        14: [[2, 3]],
        15: []
    }

    segments = []
    # we iterate through every 2x2 cell in the grid
    for row in range(h - 1):
        for col in range(w - 1):
            # values for the 4 corners of the current cell
            top_left = depth_map[row, col]
            top_right = depth_map[row, col + 1]
            bottom_left = depth_map[row + 1, col]
            bottom_right = depth_map[row + 1, col + 1]

            # compare with isovalue to tell which values are included inside
            tl_inside = top_left >= isovalue
            tr_inside = top_right >= isovalue
            bl_inside = bottom_left >= isovalue
            br_inside = bottom_right >= isovalue

            # calculate the lookup index
            state_index = (tl_inside * 8) + (tr_inside * 4) + (
                        br_inside * 2) + (bl_inside * 1)

            # solve the saddle cases with 5 and 10 using center averages
            if state_index == 5 or state_index == 10:
                average = (
                                      top_left + top_right + bottom_left + bottom_right) / 4
                if average >= isovalue:
                    edges_to_connect = states[state_index]["center_high"]
                else:
                    edges_to_connect = states[state_index]["center_low"]

            # otherwise just look up directly
            else:
                edges_to_connect = states[state_index]

            # coordinate guide:
            top = (col + 0.5, row)
            right = (col + 1, row + 0.5)
            bottom = (col + 0.5, row + 1)
            left = (col, row + 0.5)

            edge_coords = [top, right, bottom, left]

            # generate and store line segments by connecting the target edges
            for edge1, edge2 in edges_to_connect:
                p1 = edge_coords[edge1]
                p2 = edge_coords[edge2]
                segments.append((p1, p2))

    return segments

