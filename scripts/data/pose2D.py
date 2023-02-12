# standard
import math

# 3rd party
import numpy


def normalization(Xx, Xy):
  T, n = Xx.shape
  sum0 = T * n
  sum1Xx = numpy.sum(numpy.sum(Xx))
  sum2Xx = numpy.sum(numpy.sum(Xx * Xx))
  sum1Xy = numpy.sum(numpy.sum(Xy))
  sum2Xy = numpy.sum(numpy.sum(Xy * Xy))
  mux = sum1Xx / sum0
  muy = sum1Xy / sum0
  sum0 = 2 * sum0
  sum1 = sum1Xx + sum1Xy
  sum2 = sum2Xx + sum2Xy
  mu = sum1 / sum0
  sigma2 = (sum2 / sum0) - mu * mu
  if sigma2 < 1e-10:
    simga2 = 1e-10
  sigma = math.sqrt(sigma2)
  return (Xx - mux) / sigma, (Xy - muy) / sigma
      

import numpy as np

import numpy as np

# def prune_keypoints_weighted_average(x_points, y_points, probabilities, threshold):
#     N = len(x_points)
#     Xx = np.array(x_points).reshape((1, N))
#     Xy = np.array(y_points).reshape((1, N))
#     Xw = np.array(probabilities).reshape((1, N))
#     Yx = np.zeros_like(Xx)
#     Yy = np.zeros_like(Xy)
#     Yw = np.zeros_like(Xw)
#     T = Xx.shape[0]
#     watchThis = range(N)
#     dtype = np.float32
#     for t in range(T):
#         sum0 = 0
#         sum1 = 0.0
#         for i in watchThis:
#             sum0 = sum0 + 1
#             sum1 = sum1 + Xw[t, i]
#         Ew = sum1 / sum0
#         if Ew >= threshold:
#             for i in range(N):
#                 Yx[t, i] = Xx[t, i]
#                 Yy[t, i] = Xy[t, i]
#                 Yw[t, i] = Xw[t, i]
#     pruned_x_points = Yx.flatten().tolist()
#     pruned_y_points = Yy.flatten().tolist()
#     pruned_probabilities = Yw.flatten().tolist()
#     return pruned_x_points, pruned_y_points, pruned_probabilities


import numpy as np
import numpy as np
import numpy as np

def keypoint_interpolation(x_points, y_points, probabilities, threshold):
    N = len(x_points)
    T = len(x_points[0])
    Yx = np.zeros((T, N))
    Yy = np.zeros((T, N))
    for t in range(T):
        for i in range(N):
            a1 = x_points[i][t]
            a2 = y_points[i][t]
            p = probabilities[i][t]
            sumpa1 = p * a1
            sumpa2 = p * a2
            sump = p
            delta = 0
            while sump < threshold:
                change = False
                delta = delta + 1
                t2 = t + delta
                if t2 < T:
                    a1 = x_points[i][t2]
                    a2 = y_points[i][t2]
                    p = probabilities[i][t2]
                    sumpa1 = sumpa1 + p * a1
                    sumpa2 = sumpa2 + p * a2
                    sump = sump + p
                    change = True
                t2 = t - delta
                if t2 >= 0:
                    a1 = x_points[i][t2]
                    a2 = y_points[i][t2]
                    p = probabilities[i][t2]
                    sumpa1 = sumpa1 + p * a1
                    sumpa2 = sumpa2 + p * a2
                    sump = sump + p
                    change = True
                if not change:
                    break
            if sump <= 0.0:
                sump = 1e-10
            Yx[t, i] = sumpa1 / sump
            Yy[t, i] = sumpa2 / sump
    return Yx, Yy



def prune(Xx, Xy, Xw, watchThis, threshold, dtype):
  T = Xw.shape[0]
  N = Xw.shape[1]
  Yx = numpy.zeros((T, N), dtype=dtype)
  Yy = numpy.zeros((T, N), dtype=dtype)
  Yw = numpy.zeros((T, N), dtype=dtype)
  for t in range(T):
    sum0 = 0
    sum1 = 0.0
    for i in watchThis:
      sum0 = sum0 + 1
      sum1 = sum1 + Xw[t, i]
    Ew = sum1 / sum0
    if Ew >= threshold:
      for i in range(N):
        Yx[t, i] = Xx[t, i]
        Yy[t, i] = Xy[t, i]
        Yw[t, i] = Xw[t, i]
  return Yx, Yy, Yw


def interpolation(Xx, Xy, Xw, threshold, dtype):
  T = Xw.shape[0]
  N = Xw.shape[1]
  Yx = numpy.zeros((T, N), dtype=dtype)
  Yy = numpy.zeros((T, N), dtype=dtype)
  for t in range(T):
    for i in range(N):
      a1 = Xx[t, i]
      a2 = Xy[t, i]
      p = Xw[t, i]
      sumpa1 = p * a1
      sumpa2 = p * a2
      sump = p
      delta = 0
      while sump < threshold:
        change = False
        delta = delta + 1
        t2 = t + delta
        if t2 < T:
          a1 = Xx[t2, i]
          a2 = Xy[t2, i]
          p = Xw[t2, i]
          sumpa1 = sumpa1 + p * a1
          sumpa2 = sumpa2 + p * a2
          sump = sump + p
          change = True
        t2 = t - delta
        if t2 >= 0:
          a1 = Xx[t2, i]
          a2 = Xy[t2, i]
          p = Xw[t2, i]
          sumpa1 = sumpa1 + p * a1
          sumpa2 = sumpa2 + p * a2
          sump = sump + p
          change = True
        if not change:
          break
      if sump <= 0.0:
        sump = 1e-10
      Yx[t, i] = sumpa1 / sump
      Yy[t, i] = sumpa2 / sump
  return Yx, Yy, Xw



def prune_keypoints_weighted_average(coords, threshold, base_thresh):
  threshold = numpy.maximum(threshold, base_thresh)
  T, N, _ = coords.shape
  x_points = coords[:, :, 0].flatten()
  y_points = coords[:, :, 1].flatten()
  probabilites = threshold.flatten()

  valid = probabilites > 0

  x_points = x_points[valid]
  y_points = y_points[valid]
  probabilites = probabilites[valid]

  weights = probabilites / numpy.sum(probabilites)
  x_point = numpy.sum(x_points * weights)
  y_point = numpy.sum(y_points * weights)

  return x_point, y_point, probabilites.mean()


import numpy as np

def interpolate_keypoints(coords, threshold):
    T, N, _ = coords.shape
    for t in range(T):
        for n in range(N):
            if threshold[t, n] < 0.2:
                nan_indices = np.where(np.isnan(coords[t, n, 0]))[0]
                if len(nan_indices) > 0:
                    prev_valid_index = nan_indices[nan_indices < t][-1] if nan_indices[nan_indices < t].size > 0 else -1
                    next_valid_index = nan_indices[nan_indices > t][0] if nan_indices[nan_indices > t].size > 0 else T
                    coords[t, n, 0] = (coords[prev_valid_index, n, 0] + coords[next_valid_index, n, 0]) / 2
                    coords[t, n, 1] = (coords[prev_valid_index, n, 1] + coords[next_valid_index, n, 1]) / 2
    return coords



import numpy as np

import numpy as np

def prune_keypoints(coords, threshold, base_threshold):
    T, N, _ = coords.shape
    new_coords = np.copy(coords)
    for t in range(T):
        for n in range(N):
            if threshold[t][n] < base_threshold:
                new_coords[t][n] = [np.NaN, np.NaN]
    return new_coords

