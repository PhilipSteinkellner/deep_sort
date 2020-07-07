# vim: expandtab:ts=4:sw=4
import numpy as np
import math


def distance(bbox, candidates):
    """Compute distance between bbox and candidates.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding eixboxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The distance in [0, 1] between the `bbox` and each
        candidate. A higher score means that 'bbox' and 'candidate'
        are further apart.

    """
    # compute center for bbox and candidates (x and y coordinates)
    center_bbox = [bbox[0] + bbox[2]/2, bbox[1] - bbox[3]/2]
    center_candidates = []
    for c in candidates:
        center_candidates.append([c[0] + c[2]/2, c[1] - c[3]/2])
    
    distances = []
    for center_candidate in center_candidates:
        # use euclidean distance
        distances.append(math.sqrt(math.pow(center_bbox[0] - center_candidate[0], 2)
                                   + math.pow(center_bbox[1] - center_candidate[1], 2)))
    return np.array(distances)

def distance_cost(video_dimensions, tracks, detections, track_indices=None,
             detection_indices=None):
    """A distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - distance(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))
        
    max_distance = math.sqrt(math.pow(video_dimensions[0], 2) + math.pow(video_dimensions[1], 2))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        #if tracks[track_idx].time_since_update > 1:
        #    cost_matrix[row, :] = linear_assignment.INFTY_COST
        #    continue
        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = distance(bbox, candidates) / max_distance
    return cost_matrix
