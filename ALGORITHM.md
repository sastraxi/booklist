
  # algorithm as follows:
  # - group lines together by "best adjacency" (x adj to y iff dot(x_vec, y_yec) < 0.15);
  #   - minimize min. distance between line endpoints (?? while maximizing segment length ??)
  # - join those lines by extending them in the direction of their closest endpoints until they hit and share an endpoint
  # - make rectangles by completing the two lines, either by:
  #   - finding another line (closest to the length of the other side) that matches up well enough, or
  #   - completing the rectangle naively (just mirro the segments)
  # - remove rectangles that are >= 80% inside other rectangles
  # - 
