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
    diff = cv2.absdiff(image, np.uint8(bg_color))
    diff_sum = np.sum(diff, axis=2)
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
    # Choose the largest contour (by area)
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype="float32")
    ordered_box = order_points(box)
    return ordered_box, contours


def flood_fill_blackout(image, loDiff=(30, 30, 30), upDiff=(30, 30, 30)):
    """
    Performs flood fill from each of the four corners on the given image.
    Regions connected to the image border that are within the specified
    difference range are filled with black.

    Returns the image with background regions blacked out.
    """
    filled = image.copy()
    h, w = filled.shape[:2]
    # OpenCV's floodFill requires a mask 2 pixels larger than the image.
    mask = np.zeros((h+2, w+2), np.uint8)
    # Define seed points: the four corners.
    seed_points = [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]
    for seed in seed_points:
        cv2.floodFill(filled, mask, seed, (0, 0, 0), loDiff,
                      upDiff, flags=cv2.FLOODFILL_FIXED_RANGE)
    return filled


def refine_extraction(extracted):
    """
    Refines the extracted image by thresholding and using morphological erosion to
    remove soft shadow borders. Then finds the largest contour and crops to its bounding box.
    """
    # Convert to grayscale.
    gray = cv2.cvtColor(extracted, cv2.COLOR_BGR2GRAY)
    # Use Otsu thresholding to separate foreground from shadows.
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Erode the threshold mask to remove soft edges.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    eroded = cv2.erode(thresh, kernel, iterations=1)

    # Find contours in the eroded image.
    contours, _ = cv2.findContours(
        eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return extracted  # Fallback: return original extraction.

    # Assume the largest contour is the photo.
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Crop the extracted image using the bounding box.
    refined = extracted[y:y+h, x:x+w]
    return refined


def main():
    # Get the image file from command-line or use a default.
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "sample.jpg"

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from '{image_path}'")
        sys.exit(1)

    # Determine the background color from the corners.
    bg_color = get_background_color(image)
    # Create a mask from the background.
    mask = create_mask_from_background(image, bg_color, tolerance=100)
    refined_mask = refine_mask(mask, kernel_size=7)

    # Extract the photo boundary using the refined mask.
    box, contours = extract_photo(image, refined_mask)
    if box is None:
        print("No photo detected.")
        sys.exit(1)

    # Apply perspective transform to extract the photo.
    extracted = four_point_transform(image, box)

    # Use flood fill to black out background areas on the extracted image.
    extracted_black = flood_fill_blackout(
        extracted, loDiff=(30, 30, 30), upDiff=(30, 30, 30))

    # Now refine the extraction to tighten up the edges.
    refined_extraction = refine_extraction(extracted_black)

    # Draw all detected contours (blue) and the chosen boundary (green) on a copy.
    contours_drawn = image.copy()
    cv2.drawContours(contours_drawn, contours, -1, (255, 0, 0), 2)
    cv2.polylines(contours_drawn, [box.astype(int)], True, (0, 255, 0), 3)

    # Save the outputs.
    cv2.imwrite("refined_mask.jpg", refined_mask)
    cv2.imwrite("detected_contours.jpg", contours_drawn)
    cv2.imwrite("extracted_photo.jpg", extracted)
    cv2.imwrite("extracted_photo_black.jpg", extracted_black)
    cv2.imwrite("extracted_photo_refined.jpg", refined_extraction)

    print("Saved 'refined_mask.jpg', 'detected_contours.jpg', 'extracted_photo.jpg', and 'extracted_photo_black.jpg'.")


if __name__ == "__main__":
    main()
