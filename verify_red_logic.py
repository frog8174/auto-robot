import cv2
import numpy as np
import argparse
import sys

def check_red(frame, debug=False):
    if frame is None: return False
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Red has two ranges in HSV (0-10 and 170-180)
    # Range 1
    lower1 = np.array([0, 70, 50])
    upper1 = np.array([10, 255, 255])
    
    # Range 2
    lower2 = np.array([170, 70, 50])
    upper2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    full_mask = mask1 + mask2
    
    # Dilate/Erode to remove noise (Morphological Opening)
    kernel = np.ones((3,3), np.uint8)
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    count = cv2.countNonZero(full_mask)
    h, w = frame.shape[:2]
    total_pixels = h * w
    ratio = count / total_pixels
    
    print(f"[INFO] Red Pixel Ratio: {ratio*100:.2f}% (Threshold: 5.00%)")
    
    if debug:
        cv2.imshow("Original", frame)
        cv2.imshow("Red Mask", full_mask)
        print("Press any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # If > 5% of screen is red, consider it "Large Area"
    return ratio > 0.05

def create_dummy_image(color, width=640, height=480):
    # Color in BGR
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = color
    return img

def main():
    parser = argparse.ArgumentParser(description='Test Red Detection Logic on PC')
    parser.add_argument('image_path', nargs='?', help='Path to an image file')
    args = parser.parse_args()

    if args.image_path:
        # Load user provided image
        print(f"Loading image: {args.image_path}")
        frame = cv2.imread(args.image_path)
        if frame is None:
            print(f"Error: Could not read image at {args.image_path}")
            sys.exit(1)
        
        result = check_red(frame, debug=True)
        print(f"Result: {'RED DETECTED' if result else 'NO RED DETECTED'}")
        
    else:
        print("No image path provided. Running synthetic tests...")
        
        # Test 1: Pure Red Image
        print("\n--- Test 1: Pure Red Image ---")
        red_img = create_dummy_image((0, 0, 255)) # BGR
        result_red = check_red(red_img, debug=False)
        print(f"Expected: True, Got: {result_red}")
        
        # Test 2: Pure Blue Image
        print("\n--- Test 2: Pure Blue Image ---")
        blue_img = create_dummy_image((255, 0, 0)) # BGR
        result_blue = check_red(blue_img, debug=False)
        print(f"Expected: False, Got: {result_blue}")

        # Test 3: 10% Red Patch
        print("\n--- Test 3: 10% Red Patch on Black Background ---")
        patch_img = create_dummy_image((0, 0, 0))
        h, w = patch_img.shape[:2]
        # Fill top 10% with red
        patch_img[0:int(h*0.1), :] = (0, 0, 255)
        result_patch = check_red(patch_img, debug=False)
        print(f"Expected: True, Got: {result_patch}")

        # Test 4: 2% Red Patch
        print("\n--- Test 4: 2% Red Patch on Black Background ---")
        small_patch_img = create_dummy_image((0, 0, 0))
        # Fill top 2% with red
        small_patch_img[0:int(h*0.02), :] = (0, 0, 255)
        result_small = check_red(small_patch_img, debug=False)
        print(f"Expected: False, Got: {result_small}")

if __name__ == "__main__":
    main()
