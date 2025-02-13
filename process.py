import cv2
import numpy as np
import sys


def get_background_color(image, sample_size=10):
    """
    Samples small patches from the four corners of the image and returns
    the average background color.
    """
    h, w = image.shape[:2]
    patches = [
        image[0:sample_size, 0:sample_size],
        image[0:sample_size, w-sample_size:w],
        image[h-sample_size:h, 0:sample_size],
        image[h-sample_size:h, w-sample_size:w]
    ]
    patches = np.concatenate(patches, axis=0)
    mean_color = np.mean(patches, axis=(0, 1))
    return mean_color


def create_mask_from_background(image, bg_color, tolerance=20):
    """
    Creates a binary mask where pixels that differ from the background color
    by more than the tolerance (across all channels) are considered foreground.
    """
    # Compute per-pixel absolute difference from the background color.
    diff = cv2.absdiff(image, np.uint8(bg_color))
    diff_sum = np.sum(diff, axis=2)
    # Pixels with total difference greater than tolerance become foreground.
    mask = np.uint8(diff_sum > tolerance) * 255
    return mask


def refine_mask(mask, kernel_size=7):
    """
    Uses morphological operations to close small holes and remove noise.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened


def order_points(pts):
    """
    Orders an array of four points in the order:
    top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    """
    Warps the image so that the region defined by pts becomes a top-down view.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # Compute width of new image.
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    # Compute height of new image.
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    # Destination coordinates.
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    # Compute perspective transform matrix and warp.
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def extract_photo(image, mask):
    """
    Finds contours in the mask, selects the largest one, and uses a rotated
    rectangle (minAreaRect) to approximate its boundary.
    Returns the ordered 4 points of the rectangle and all detected contours.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    # Choose the largest contour (by area).
    largest = max(contours, key=cv2.contourArea)
    # Obtain the rotated rectangle.
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype="float32")
    ordered_box = order_points(box)
    return ordered_box, contours


def refine_extraction_by_border_scan(img, std_threshold=10):
    """
    Starting from each edge of the image, scans row-by-row (or column-by-column)
    until the pixel intensity standard deviation exceeds std_threshold, indicating
    the start of photo content. Crops the image to the detected bounds.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Scan from top.
    top = 0
    for i in range(h):
        if np.std(gray[i, :]) > std_threshold:
            top = i
            break

    # Scan from bottom.
    bottom = h - 1
    for i in range(h - 1, -1, -1):
        print("Bottom std:", np.std(gray[i, :]))
        if np.std(gray[i, :]) > std_threshold:
            bottom = i
            break

    # Scan from left.
    left = 0
    for j in range(w):
        if np.std(gray[:, j]) > std_threshold:
            left = j
            break

    # Scan from right.
    right = w - 1
    for j in range(w - 1, -1, -1):
        if np.std(gray[:, j]) > std_threshold:
            right = j
            break

    if top >= bottom or left >= right:
        return img  # Return original if scanning fails.
    refined = img[top:bottom+1, left:right+1]
    return refined


def main():
    # Get image file from command-line or use a default.
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "sample.jpg"

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from '{image_path}'")
        sys.exit(1)

    # Sample background color from the corners.
    bg_color = get_background_color(image)
    # Create and refine the foreground mask.
    mask = create_mask_from_background(image, bg_color, tolerance=200)
    refined_mask = refine_mask(mask, kernel_size=7)

    # Extract photo boundary using the refined mask.
    box, contours = extract_photo(image, refined_mask)
    if box is None:
        print("No photo detected.")
        sys.exit(1)

    # Apply perspective transform to extract the photo.
    extracted = four_point_transform(image, box)

    # Now scan inward to detect where the photo content begins.
    refined = refine_extraction_by_border_scan(extracted, std_threshold=20)
    cv2.imwrite("extracted_refined.jpg", refined)

    # Draw all detected contours (in blue) and the chosen boundary (in green).
    contours_drawn = image.copy()
    cv2.drawContours(contours_drawn, contours, -1, (255, 0, 0), 2)
    cv2.polylines(contours_drawn, [box.astype(int)], True, (0, 255, 0), 3)

    # Save the results.
    cv2.imwrite("refined_mask.jpg", refined_mask)
    cv2.imwrite("detected_contours.jpg", contours_drawn)
    cv2.imwrite("extracted_photo.jpg", extracted)

    print("Saved 'refined_mask.jpg', 'detected_contours.jpg', and 'extracted_photo.jpg'.")


if __name__ == "__main__":
    main()
