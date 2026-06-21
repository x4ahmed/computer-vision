import numpy as np
import cv2
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


def get_affine_transformation(points_in, points_out):
    """
    Estimate the affine transformation matrix mapped from points_in to points_out.
    Transform the input points to homogenous coordinates and solve the least-squares problem.
    """
    # transform to homogenous coordinates by adding a column of ones
    ones = np.ones((points_in.shape[0], 1))
    A = np.hstack([points_in, ones])  # shape: (N, 3)

    # solve the least-squares problem for x and y separately
    # A @ [a11, a12, tx] = x_out
    # A @ [a21, a22, ty] = y_out
    x_params, _, _, _ = np.linalg.lstsq(A, points_out[:, 0], rcond=None)
    y_params, _, _, _ = np.linalg.lstsq(A, points_out[:, 1], rcond=None)

    # build the 2x3 affine matrix
    matrix = np.array([
        [x_params[0], x_params[1], x_params[2]],
        [y_params[0], y_params[1], y_params[2]]
    ])
    return matrix


def transform_points(points, matrix):
    """
    Given a set of 2D points, apply the transformation matrix.
    Return the new (x, y) coordinates.
    """
    # convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])  # shape: (N, 3)

    # transform the points
    transformed = points_h @ matrix.T  # shape: (N, 2)
    return transformed


def backwards_mapping(image, output_shape, transformation, background=0):
    """
    Apply a backward mapping transformation to the input image.
    For each pixel in the output image, find where it came from in the input image.
    """
    # create the coordinates of the output image
    y_coords, x_coords = np.indices(output_shape)
    output_coords = np.stack([y_coords.ravel(), x_coords.ravel()], axis=-1)

    # transform the output coordinates back into the original image space
    src_coords = transform_points(output_coords, transformation)
    src_y = src_coords[:, 0].reshape(output_shape)
    src_x = src_coords[:, 1].reshape(output_shape)

    # use bilinear interpolation to sample pixel values
    output_image = np.full(output_shape, background, dtype=image.dtype)

    # get integer coordinates and fractional parts
    x0 = np.floor(src_x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(src_y).astype(int)
    y1 = y0 + 1

    # clip to image bounds
    x0 = np.clip(x0, 0, image.shape[1] - 1)
    x1 = np.clip(x1, 0, image.shape[1] - 1)
    y0 = np.clip(y0, 0, image.shape[0] - 1)
    y1 = np.clip(y1, 0, image.shape[0] - 1)

    # bilinear interpolation weights
    wa = (x1 - src_x) * (y1 - src_y)
    wb = (src_x - x0) * (y1 - src_y)
    wc = (x1 - src_x) * (src_y - y0)
    wd = (src_x - x0) * (src_y - y0)

    # sample and interpolate
    output_image = (
        wa * image[y0, x0] +
        wb * image[y0, x1] +
        wc * image[y1, x0] +
        wd * image[y1, x1]
    )

    # mark out-of-bounds pixels as background
    valid = (
        (src_x >= 0) & (src_x < image.shape[1] - 1) &
        (src_y >= 0) & (src_y < image.shape[0] - 1)
    )
    output_image = np.where(valid, output_image, background)

    return output_image


def downsample_bilinear(img, factor):
    """
    Downsample a grayscale image by the given factor using bilinear interpolation.
    """
    new_h = int(img.shape[0] / factor)
    new_w = int(img.shape[1] / factor)

    # create grid of coordinates in the output image
    y = np.linspace(0, img.shape[0] - 1, new_h)
    x = np.linspace(0, img.shape[1] - 1, new_w)
    xv, yv = np.meshgrid(x, y)

    # get integer and fractional parts
    x0 = np.floor(xv).astype(int)
    x1 = np.clip(x0 + 1, 0, img.shape[1] - 1)
    y0 = np.floor(yv).astype(int)
    y1 = np.clip(y0 + 1, 0, img.shape[0] - 1)

    x0 = np.clip(x0, 0, img.shape[1] - 1)
    y0 = np.clip(y0, 0, img.shape[0] - 1)

    # interpolation weights
    wa = (x1 - xv) * (y1 - yv)
    wb = (xv - x0) * (y1 - yv)
    wc = (x1 - xv) * (yv - y0)
    wd = (xv - x0) * (yv - y0)

    # sample and interpolate
    downsampled = (
        wa * img[y0, x0] +
        wb * img[y0, x1] +
        wc * img[y1, x0] +
        wd * img[y1, x1]
    )

    return downsampled


def histogram_equalization(img):
    """
    Compute histogram equalization mapping from an image.
    Return the equalized image.
    """
    # compute histogram
    hist, bins = np.histogram(img.flatten(), bins=256, range=(0, 256))

    # compute cumulative distribution function (CDF)
    cdf = hist.cumsum()

    # normalize CDF to [0, 255]
    cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0
    cdf_normalized = ((cdf - cdf_min) / (cdf.max() - cdf_min) * 255).astype(np.uint8)

    # map original pixel values to equalized values
    equalized = cdf_normalized[img.astype(int)]

    return equalized


def convolve2d(img, kernel):
    """
    Apply a 2D convolution (without padding, assumes odd kernel).

    Parameters:
        img (np.ndarray): Grayscale image.
        kernel (np.ndarray): 2D filter kernel.

    Returns:
        np.ndarray: Convolved image (same size as input, zero-padded).
    """
    kernel_h, kernel_w = kernel.shape
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2

    # pad the image with zeros
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    output = np.zeros_like(img, dtype=np.float64)

    # slide the kernel over the image
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i + kernel_h, j:j + kernel_w]
            output[i, j] = np.sum(region * kernel)

    return output


def sobel_filter(img):
    """
    Apply Sobel edge detection filter (magnitude of gradients).

    Parameters:
        img (np.ndarray): Grayscale image.

    Returns:
        np.ndarray: Sobel gradient magnitude image.
    """
    # Sobel kernels for x and y gradients
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    # compute gradients
    grad_x = convolve2d(img, sobel_x)
    grad_y = convolve2d(img, sobel_y)

    # compute magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    return magnitude


def filter_wrapper_fn(img, mode, kernel_size=3, sigma=1.0):
    """
    Apply one of the following filters to the image: mean, median, gaussian, sobel.

    Parameters:
        img (np.ndarray): Grayscale image.
        mode (str): One of "mean", "median", "gaussian", "sobel"
        kernel_size (int): Kernel size (must be odd)
        sigma (float): Gaussian std dev (only used for gaussian)

    Returns:
        np.ndarray: Filtered image.
    """
    if mode == "mean":
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        return convolve2d(img, kernel)

    elif mode == "median":
        pad = kernel_size // 2
        padded = np.pad(img, ((pad, pad), (pad, pad)), mode='edge')
        output = np.zeros_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                region = padded[i:i + kernel_size, j:j + kernel_size]
                output[i, j] = np.median(region)
        return output

    elif mode == "gaussian":
        # create Gaussian kernel
        ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / np.sum(kernel)
        return convolve2d(img, kernel)

    elif mode == "sobel":
        return sobel_filter(img)

    else:
        raise ValueError(f"Unknown filter mode: {mode}")


def get_keypoints(image, filtering=True, sigma=3):
    """
    Extracts keypoints from the image using SIFT.
    Optionally smooth the image with a gaussian filter first.
    """
    if filtering:
        image = gaussian_filter(image, sigma=sigma)

    # convert to uint8 if needed for OpenCV SIFT
    if image.dtype != np.uint8:
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    return keypoints, descriptors


def intersect2d(array1, array2):
    """ Helper to get intersection of row matches """
    test = array1[:, None] == array2
    return array2[np.all(test.mean(0) > 0, axis=1)]


def matching(descriptors_1, descriptors_2, max_ratio=0.7, cross_checking=True):
    """
    Matches the descriptors against each other.
    Returns the best match for each descriptor, if it is significant.
    The significance is defined by the max_ratio: distance_1 / distance_2 < max_ratio.
    Optional cross-checking of matches.
    """
    # compute pairwise distances between all descriptors
    distances = cdist(descriptors_1, descriptors_2, metric='euclidean')

    # get the two smallest distances for each descriptor in descriptors_1
    # using argpartition for efficiency
    partitioned = np.argpartition(distances, 1, axis=1)[:, :2]
    best_distances = np.take_along_axis(distances, partitioned, axis=1)

    # compute ratio test: distance to best match / distance to second best match
    ratios = best_distances[:, 0] / (best_distances[:, 1] + 1e-10)

    # keep matches where the best match is significantly better than the second best
    mask1 = ratios < max_ratio
    matches1 = np.stack((np.arange(descriptors_1.shape[0])[mask1], partitioned[mask1, 0]), axis=1)

    if not cross_checking:
        return matches1

    # cross-checking: do the same from descriptors_2 to descriptors_1
    partitioned2 = np.argpartition(distances, 1, axis=0)[:2, :]
    best_distances2 = np.take_along_axis(distances, partitioned2, axis=0)

    ratios2 = best_distances2[0, :] / (best_distances2[1, :] + 1e-10)
    mask2 = ratios2 < max_ratio
    final_matches2 = np.stack(
        (np.arange(descriptors_2.shape[0])[mask2], partitioned2[0, mask2]),
    ).T

    # return the intersection of the two matches arrays
    # invert final_matches2 to point in the same direction
    final_matches = intersect2d(matches1, final_matches2[:, ::-1])

    return final_matches


def ransac(points_in, points_out, matches, percentage_outliers=0.5, probability=0.99, cutoff=20, k=3):
    """
    Implement Random Sample Consensus to predict a robust affine model on a set with outliers.
    """
    # get matched point pairs
    matched_in = points_in[matches[:, 0]]
    matched_out = points_out[matches[:, 1]]

    n_points = matched_in.shape[0]

    # calculate number of iterations needed
    # probability = 1 - (1 - (1 - p)^k)^N
    # solving for N: N = log(1 - probability) / log(1 - (1 - p)^k)
    if percentage_outliers >= 1.0:
        n_iterations = 1
    else:
        n_iterations = int(
            np.ceil(
                np.log(1 - probability) /
                np.log(1 - (1 - percentage_outliers) ** k)
            )
        )

    inlier_best = np.array([], dtype=int)
    support_best = 0

    for _ in range(n_iterations):
        # randomly sample k points
        indices = np.random.choice(n_points, k, replace=False)
        sample_in = matched_in[indices]
        sample_out = matched_out[indices]

        # compute affine transformation from sample
        try:
            transform = get_affine_transformation(sample_in, sample_out)
        except np.linalg.LinAlgError:
            continue

        # apply transformation to all points
        predicted = transform_points(matched_in, transform)

        # compute residuals (Euclidean distance)
        residuals = np.linalg.norm(predicted - matched_out, axis=1)

        # count inliers
        inliers = residuals < cutoff
        support = np.sum(inliers)

        # update best model if this one has more support
        if support > support_best:
            support_best = support
            inlier_best = inliers

    # return the inliers
    return matches[inlier_best], support_best


def helper_plot_fn(img1, img2, transformed_img2):
    """ Plotting helper """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title("Source Image")
    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title("Destination Image")
    axes[2].imshow(transformed_img2, cmap='gray')
    axes[2].set_title("Transformed Image")
    plt.tight_layout()
    plt.show()
