## EdgeEnhance

### Overview

**EdgeEnhance** is an edge and boundary detection toolkit that integrates **Canny edge detection**, **Hough Transform**, and **Active Contour Models (Snakes)** to extract and analyze object boundaries in grayscale and color images. The system identifies structural shapes (lines, circles, ellipses) and evolves contours around object regions using greedy snake algorithms.

> This project bridges classical edge detection with deformable models to extract structural and anatomical features from real-world images — from CT scans to natural scenes.

![EdgeEnhance Overview](https://github.com/user-attachments/assets/00d25199-949f-4fe5-b372-7c593e2dcccc)


---

### Features & Visual Examples

---

#### Canny Edge Detection

<table>
<tr>
<td><b>Original Image</b></td>
<td><b>Canny Edge Map</b></td>
</tr>
<tr>
<td><img src="https://github.com/user-attachments/assets/8aefa4b8-8738-4b62-a361-d81981b208d3" width="250" height= "250"/></td>
<td><img src="https://github.com/user-attachments/assets/ae02ba32-ab51-4bbc-82f6-55cb9797a06e" width="250" height= "250"/></td>
</tr>
</table>

> **Insight:** The Canny detector captures fine-grained structural edges by computing gradients, applying non-maximum suppression, and executing hysteresis thresholding.
>
> **Parameters used:**
>
> * **Gaussian Kernel Size:** 5×5 (for smoother noise reduction)
> * **Sigma (σ):** 1.2 (ideal for natural images)
> * **High Threshold:** 100
> * **Low Threshold:** 20
> * **Sobel Kernel Size:** 3×3
> * **Gradient Method:** Manhattan Distance (used here for higher edge contrast)

---

#### Hough Transform – Shape Detection

Applies classical Hough Transform to detect geometric structures in edge maps using accumulator-based voting. The following shapes are detected independently:

---

##### Hough Line Detection

<table>
<tr>
<td><b>Original Image</b></td>
<td><b>Detected Lines</b></td>
</tr>
<tr>
<td><img src="https://github.com/user-attachments/assets/c87ec1b3-36b1-4d31-811d-dd7ecf8065ec" width="250" height = "250"/></td>
<td><img src="https://github.com/user-attachments/assets/5cf747ac-88a3-4a1d-b260-bd0603319438" width="250" height = "250"/></td>
</tr>
</table>

> **Insight:** Hough Line Transform detects straight edges by identifying colinear points in gradient space. A threshold of 300 votes was applied to ensure only dominant structural lines are retained. Useful for architectural and structural boundaries.
>
> **Parameter used:**
>
> * **Line Threshold (Votes):** 300

---

##### Hough Circle Detection

<table>
<tr>
<td><b>Original Image</b></td>
<td><b>Detected Circles</b></td>
</tr>
<tr>
<td><img src="https://github.com/user-attachments/assets/a9f66489-c59b-409f-a680-7df9fbf9a154" width="250"/></td>
<td><img src="https://github.com/user-attachments/assets/8769110d-3f6c-460c-8ea4-6e4c4bac6e5d" width="250"/></td>
</tr>
</table>

> **Insight:** Circle detection works by mapping edge pixels into a 3D parameter space (center X, center Y, radius).
> This method is robust to partial contours and is ideal for identifying rounded anatomical or artificial shapes.
>
> **Parameters used:**
>
> * **Minimum Radius:** 10
> * **Maximum Radius:** 60
> * **Canny Threshold:** 200
> * **Accumulator Threshold:** 90

---

##### Hough Ellipse Detection

<table>
<tr>
<td><b>Original Image</b></td>
<td><b>Detected Ellipses</b></td>
</tr>
<tr>
<td><img src="https://github.com/user-attachments/assets/8f6ac26f-856b-479e-990a-2d70dc78be88" width="250" height = "250"/></td>
<td><img src="https://github.com/user-attachments/assets/b3d04dca-bc82-4ccc-8c27-19d95003540a" width="250" height = "250"/></td>
</tr>
</table>

> **Insight:** Ellipse detection is more complex due to its 5D parameter space (center coordinates, major and minor axes, and orientation).
> It enables flexible shape fitting for biological structures and irregular contours.
>
> **Parameters used:**
>
> * **Minimum Ellipse Length:** 40
> * **Maximum Ellipse Length:** 50
> * **Ellipse Threshold (votes):** 95

---

#### Active Contour Model (Snake)

<table>
<tr>
<td><b>Original Image</b></td>
<td><b>Snake Contour</b></td>
</tr>
<tr>
<td><img src="https://github.com/user-attachments/assets/d99523eb-9058-41a4-b46a-4956fb8e1103" width="250" height = "250"/></td>
<td><img src="https://github.com/user-attachments/assets/698120ec-ec83-4ad8-b945-90dfef393c36" width="250" height = "250"/></td>
</tr>
</table>

> **Insight:** The Snake model dynamically adjusts to edge gradients by minimizing an energy function composed of continuity, smoothness, and edge attraction.
> It is ideal for biomedical and morphological analysis. The model outputs **chain code**, **area**, and **perimeter** of the extracted shape.
>
> **Contour Metrics:**
>
> * **Area:** 20,000
> * **Perimeter:** 774
>
> **Active Contour Parameters:**
>
> * **Circle Radius:** 0
> * **Contour Type:** 1
> * **Number of Points:** 180
> * **Iterations:** 60
> * **W\_Line:** 1
> * **W\_Edge:** 8
> * **Alpha (Elasticity):** 20.00
> * **Beta (Bending):** 0.01
> * **Gamma (Step Size):** 2.00

---

### Installation

```
git clone https://github.com/YassienTawfikk/EdgeEnhance.git
cd EdgeEnhance
pip install -r requirements.txt
python main.py
```

---

### Use Cases

* Boundary extraction in CT/MRI images
* Detection of manufactured or anatomical shapes (e.g., circles, ellipses)
* Structural shape analysis in natural or urban imagery
* Measuring perimeter and area via contour evolution

---

### Contributions

<div>
  <table align="center">
    <tr>
      <td align="center">
        <a href="https://github.com/YassienTawfikk" target="_blank">
          <img src="https://avatars.githubusercontent.com/u/126521373?v=4" width="150px;" alt="Yassien Tawfik"/><br/>
          <sub><b>Yassien Tawfik</b></sub>
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/madonna-mosaad" target="_blank">
          <img src="https://avatars.githubusercontent.com/u/127048836?v=4" width="150px;" alt="Madonna Mosaad"/><br/>
          <sub><b>Madonna Mosaad</b></sub>
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/nancymahmoud1" target="_blank">
          <img src="https://avatars.githubusercontent.com/u/125357872?v=4" width="150px;" alt="Nancy Mahmoud"/><br/>
          <sub><b>Nancy Mahmoud</b></sub>
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/nariman-ahmed" target="_blank">
          <img src="https://avatars.githubusercontent.com/u/126989278?v=4" width="150px;" alt="Nariman Ahmed"/><br/>
          <sub><b>Nariman Ahmed</b></sub>
        </a>
      </td>      
    </tr>
  </table>
</div>

---
