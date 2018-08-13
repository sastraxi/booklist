from dotenv import load_dotenv

load_dotenv('.env', verbose=True)

from PIL import Image
import pytesseract
import cv2
import os
import numpy as np
import math
from scipy.spatial import cKDTree

MAX_DISTANCE = 20
MAX_DOT_PRODUCT = 0.2

COLOURS = [
  (255, 150, 0),
  (255, 0, 150),
  (0, 255, 150),
  (0, 150, 255),
  (150, 0, 255),
  (150, 255, 0),
  (255, 80, 80),
  (80, 255, 80),
  (80, 80, 255),
  (255, 180, 180),
  (180, 255, 180),
  (180, 180, 255),
  (255, 255, 25)
];

def ocr(gray):  
  # load the image as a PIL/Pillow image, apply OCR, and then delete
  # the temporary file
  filename = "{}.png".format(os.getpid())
  cv2.imwrite(filename, gray)
  text = pytesseract.image_to_string(Image.open(filename))
  os.remove(filename)
  print(text)

def dot(a, b):
  return np.dot(a, b)

def dist(pt_a, pt_b):
  x = np.subtract(pt_a, pt_b)
  return math.sqrt(dot(x, x))

def vec(segment):
  pt_a, pt_b = segment
  return np.subtract(pt_b, pt_a)

def nvec(segment):
  v = vec(segment)
  mag = np.sqrt(v.dot(v))
  return np.divide(v, mag)

def point_segment_distance_sq(pt, segment):
  pt_a, pt_b = segment
  n = vec(segment)
  pa = np.subtract(pt_a, pt)
  c = np.multiply(n, dot(n, pa) / dot(n, n))
  d = np.subtract(pa, c)
  return dot(d, d)

def close_line_indices(kdtree, point, dist, max_k=50):
  """ Returns the line indices (of parsed_lines) that are
      within dist units of the given point """
  dd, ii = kdtree.query([point], k=max_k, distance_upper_bound=dist)  
  # each segment has two endpoint indices; divide endpoint index by 2 to get segment index
  # also, cKDtree returns the sentinel value kdtree.n if fewer than "k" points found; eliminate these
  found = np.unique([ np.floor(i / 2.0) for i in ii[0] if i != kdtree.n ]).astype(int)
  return found

def line_line_isect(a, b):
  """ thanks to http://mathworld.wolfram.com/Line-LineIntersection.html """
  (x1, y1), (x2, y2) = a
  (x3, y3), (x4, y4) = b

  x_num = (x1 * y2 - x2 * y1) * (x3 - x4) - (x3 * y4 - x4 * y3) * (x1 - x2)
  y_num = (x1 * y2 - x2 * y1) * (y3 - y4) - (x3 * y4 - x4 * y3) * (y1 - y2)
  denom = (x1 - x2) * (y3 - y4) - (x3 - x4) * (y1 - y2)

  return (x_num / denom, y_num / denom)

if __name__ == "__main__":
  image_path = os.getenv("IMAGE_PATH")

  print "Loading image..."
  image = cv2.imread(image_path)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray = cv2.adaptiveThreshold(gray, 255, \
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
  gray = cv2.medianBlur(gray, 3)

  # edges = cv2.Canny(gray, 150, 255, apertureSize = 3)
  # cv2.imwrite('edges.jpg', edges)
  
  # lines = cv2.HoughLines(edges, 1, np.pi/90, 200)
  # for line in lines:
  #   for rho,theta in line:
  #       a = np.cos(theta)
  #       b = np.sin(theta)
  #       x0 = a*rho
  #       y0 = b*rho
  #       x1 = int(x0 + 1000*(-b))
  #       y1 = int(y0 + 1000*(a))
  #       x2 = int(x0 - 1000*(-b))
  #       y2 = int(y0 - 1000*(a))
  #       cv2.line(output,(x1,y1),(x2,y2),(0,0,255),4)

  # segments = cv2.HoughLinesP(edges, 1, np.pi / 45, 10, 100, 10)
  # for segment in segments:
  #   for x1,y1,x2,y2 in segment:
  #     print(x1,y1,x2,y2)
  #     cv2.line(output, (x1,y1), (x2,y2), (0,255,0), 2)

  print "Detecting lines..."
  height, width = image.shape[:2]
  overlay = image.copy()
  output = image.copy()
  alpha = 0.75
  cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
  cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

  lsd = cv2.createLineSegmentDetector(0, 0.2)
  lines = lsd.detect(gray)[0]
  parsed_lines = []
  endpoints = np.empty((lines.size / 4 * 2, 2)) # floor(index / 2) is index of line
  for n, line in enumerate(lines):
    line = line[0]
    (x1, y1, x2, y2) = line
    endpoints[2 * n] = (x1, y1)
    endpoints[2 * n + 1] = (x2, y2)
    parsed_lines.append(((x1, y1), (x2, y2)))
    
    #cv2.line(output, (x1,y1), (x2,y2), (0,255,0), 2)
  #cv2.imwrite('lines.jpg', output)

  # determine line neighbours
  print "Pairing lines..."
  kd_tree = cKDTree(endpoints)
  line_pairs = [] # list of (index_a, index_b)
  paired_indices = [] # avoid re-doing work to get symmetric result
  for this_index, (pt_a, pt_b) in enumerate(parsed_lines):
    if this_index in paired_indices:
      continue

    line_indices_a = close_line_indices(kd_tree, pt_a, MAX_DISTANCE)
    line_indices_b = close_line_indices(kd_tree, pt_a, MAX_DISTANCE)
    this_line = (pt_a, pt_b)

    lowest_score = float("inf")
    best_index = None
    for endpt, test_line_indices in [(pt_a, line_indices_a), (pt_b, line_indices_b)]:
      for test_index in test_line_indices:
        if test_index == this_index: # N.B. we allow paired_indices to show up here, so that edges can be shared
          continue

        test_line = parsed_lines[test_index]
        dist_sq = point_segment_distance_sq(endpt, test_line)
        norm_dot_product = abs(dot(nvec(this_line), nvec(test_line)))
        if norm_dot_product > MAX_DOT_PRODUCT:
          continue

        # TODO: factor in length?

        angle_score = (MAX_DOT_PRODUCT - norm_dot_product) / MAX_DOT_PRODUCT
        score = dist_sq * angle_score
        if score < lowest_score:
          lowest_score = score
          best_index = test_index

    if best_index is not None:
      # print("best pair: %s, %s" % (this_index, best_index))
      line_pairs.append((this_index, best_index))
      paired_indices.append(best_index) # don't go through outer loop with best_index (will get same result)

  # for i, (idx_a, idx_b) in enumerate(line_pairs):
  #   pt_a1, pt_a2 = parsed_lines[idx_a]
  #   pt_b1, pt_b2 = parsed_lines[idx_b]

  #   colour = COLOURS[i % len(COLOURS)]
  #   cv2.line(output, pt_a1, pt_a2, colour, 2)
  #   cv2.line(output, pt_b1, pt_b2, colour, 2)

  # cv2.imwrite('linepairs.jpg', output)
    
  # construct quadrilaterals
  # make sure lines join by extending them towards each other
  # then naively complete the quadrilateral by copying vectors from the common point
  print "Assembling quads..."
  quads = []
  for idx_a, idx_b in line_pairs:
    line_a = parsed_lines[idx_a]
    line_b = parsed_lines[idx_b]

    closest_i = None
    closest_j = None
    closest_distance = float("inf")
    for i in range(2):
      for j in range(2):
        dst = dist(line_a[i], line_b[j])
        if dst < closest_distance:
          closest_i = i
          closest_j = j
          closest_distance = dst

    # flip segments around so that the first point is the one that's
    # going to be shared (closest to the other segment)
    if closest_i == 1: line_a = (line_a[1], line_a[0])
    if closest_j == 1: line_b = (line_b[1], line_b[0])

    # determine the intersection point and use this as the corner of the quad;
    # finish the quadrilateral by extending A's vector off of B's non-shared point
    quad_a = line_a[1]
    quad_b = line_line_isect(line_a, line_b)
    quad_c = line_b[1]
    quad_d = np.add(quad_c, vec(line_a))

    # TODO: normalize quads; clockwise winding
    quads.append((quad_a, quad_b, quad_c, quad_d))

  # render quads
  alpha = 0.3
  quads_overlay = image.copy()
  cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
  for i, (pt_a, pt_b, pt_c, pt_d) in enumerate(quads):
    colour = COLOURS[i % len(COLOURS)]
    pts = np.int32([pt_a, pt_b, pt_c, pt_d])
    cv2.fillPoly(quads_overlay, [pts], colour)

  cv2.addWeighted(quads_overlay, alpha, output, 1 - alpha, 0, output)
  cv2.imwrite('quads.jpg', output)

  # extract book spines, un-distorting along the way
  # put them all in their own image for later OCR-ing
  print "Extracting spines..."
  MINIMUM_AREA = 600
  for i, (pt_a, pt_b, pt_c, pt_d) in enumerate(quads):
    width = int(dist(pt_a, pt_b)) + 1
    height = int(dist(pt_b, pt_c)) + 1
    width, height = height, width

    dst_points = np.float32([[0, height], [0, 0], [width, 0]])
    src_points = np.float32([pt_a, pt_b, pt_c])

    xform = cv2.getAffineTransform(src_points, dst_points)
    book_image = cv2.warpAffine(image, xform, (width, height))
    cv2.imwrite("quads/%s.jpg" % (i,), book_image)

  # for each rect,
