import numpy as np
import cv2
from app.processing.canny_edge import CannyEdge

class DetectCircle:

    # HAYA5OD KOL ARGUMENTS EL CANNY PLUX EL ORIGINAL IMAGE
    @staticmethod
    def superimpose(original_image, max_radius=100, min_radius=0, threshold_factor=0.7):

        # HANTALA3 DA FEL CONTROLLER TO AVOID COMPUTING EVERY TIME
        image_edges=CannyEdge.apply_canny(original_image,5,3,100,200,3,True)

        height,width=image_edges.shape
        max_radius=min(height,width)//2
        print(max_radius)
        min_radius=15

        accumulator=np.zeros((max_radius,width,height),dtype=np.uint8)

        edge_points = np.argwhere(image_edges>0)  
        angle_step=5

        angles = np.deg2rad(np.arange(0, 360, angle_step))
        # having x,y coords, and looping through r, and through angle, we want to find a,b
        for y,x in edge_points:
            for r in range(min_radius,max_radius):
                a_vals = (x - r * np.cos(angles)).astype(int)
                b_vals = (y - r * np.sin(angles)).astype(int)
                for a, b in zip(a_vals, b_vals):
                    if 0 <= a < width and 0 <= b < height:
                        accumulator[r, a, b] += 1

        threshold = np.max(accumulator) * threshold_factor # dynamic threshold
        print(threshold)
        # smaller_threshold=np.max(accumulator)*(1-0.7)

        circles = np.argwhere(accumulator>threshold)  # Get (r, a, b) where votes are high
        print(accumulator)
        print(circles)
        # smaller_circles=np.argwhere(accumulator<smaller_threshold)

        for r, a, b in circles:
            print(f"Detected circle at ({a}, {b}) with radius {r}")
            cv2.circle(original_image, (a, b), r, (0, 0, 255), 2)

        # for r, a, b in smaller_circles:
        #     print(f"Detected circle at ({a}, {b}) with radius {r}")
        #     cv2.circle(original_image, (a, b), r, (0, 255, 0), 2)

        return original_image