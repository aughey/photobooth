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
        image[0:sample_size, w - sample_size:w],
        image[h - sample_size:h, 0:sample_size],
        image[h - sample_size:h, w - sample_size:w]
    ]
    patches = np.concatenate(patches, axis=0)
    mean_color = np.mean(patches, axis=(0, 1))
    return mean_color


def create_mask_from_background(image, bg_color, tolerance=20):
    """
    Creates a binary mask where pixels that differ from the background color
    (across all channels) by more than 'tolerance' are marked as foreground.
    """
    diff = cv2.absdiff(image, np.uint8(bg_color))
    diff_sum = np.sum(diff, axis=2)
    mask = np.uint8(diff_sum > tolerance) * 255
    return mask


def refine_mask(mask, kernel_size=7):
    """
    Applies morphological operations to the mask to remove noise and fill gaps.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened


def order_points(pts):
    """
    Orders four points in the order: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]       # top-left
    rect[2] = pts[np.argmax(s)]       # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]    # top-right
    rect[3] = pts[np.argmax(diff)]    # bottom-left
    return rect


def four_point_transform(image, pts):
    """
    Performs a perspective transform to obtain a top-down view of the region
    defined by pts.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width of the new image.
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    # Compute the height of the new image.
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def extract_photo(image, refined_mask):
    """
    Finds the largest contour in the refined mask and approximates its boundary
    with a rotated rectangle. Returns the four ordered corner points.
    """
    contours, _ = cv2.findContours(
        refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype="float32")
    ordered_box = order_points(box)
    return ordered_box, contours


def flood_fill_blackout(image, loDiff=(30, 30, 30), upDiff=(30, 30, 30)):
    """
    Performs flood fill from the four corners to black out regions that are similar
    to the background.
    """
    filled = image.copy()
    h, w = filled.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    seed_points = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    for seed in seed_points:
        cv2.floodFill(filled, mask, seed, (0, 0, 0), loDiff,
                      upDiff, flags=cv2.FLOODFILL_FIXED_RANGE)
    return filled


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
    # Get image path from command-line or default to 'sample.jpg'.
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "sample.jpg"

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from '{image_path}'")
        sys.exit(1)

    # --- Background Mask Detection ---
    bg_color = get_background_color(image, sample_size=10)
    mask = create_mask_from_background(image, bg_color, tolerance=150)
    refined_mask = refine_mask(mask, kernel_size=7)
    cv2.imwrite("refined_mask.jpg", refined_mask)

    # --- Extract Photo Using the Mask ---
    box, contours = extract_photo(image, refined_mask)
    if box is None:
        print("No photo detected.")
        sys.exit(1)

    extracted = four_point_transform(image, box)
    cv2.imwrite("extracted_photo.jpg", extracted)

    # Now scan inward to detect where the photo content begins.
    refined = refine_extraction_by_border_scan(extracted, std_threshold=10)
    cv2.imwrite("extracted_refined.jpg", refined)

    # --- Visualization: Draw Detected Contours ---
    drawn = image.copy()
    cv2.drawContours(drawn, contours, -1, (255, 0, 0), 2)
    cv2.polylines(drawn, [box.astype(int)], True, (0, 255, 0), 3)
    cv2.imwrite("detected_contours.jpg", drawn)

    print("Saved output images:")
    print("  refined_mask.jpg")
    print("  extracted_photo.jpg")
    print("  extracted_photo_black.jpg")
    print("  extracted_padded.jpg")
    print("  extracted_refined.jpg")
    print("  detected_contours.jpg")


if __name__ == "__main__":
    main()
