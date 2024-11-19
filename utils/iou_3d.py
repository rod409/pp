import torch
import numpy as np
import shapely.geometry

def rotated_box_iou(boxes1, boxes2):
    """
    Calculates IoU for rotated bounding boxes.

    Args:
        boxes1 (torch.Tensor): Tensor of shape (N, 5) representing rotated boxes in format (x_center, y_center, width, height, angle).
        boxes2 (torch.Tensor): Tensor of shape (M, 5) representing rotated boxes in the same format.

    Returns:
        torch.Tensor: IoU matrix of shape (N, M).
    """

    # Convert boxes to polygons
    polygons1 = boxes_to_polygons(boxes1)
    polygons2 = boxes_to_polygons(boxes2)

    # Calculate IoU for each pair of polygons
    ious = torch.zeros((boxes1.shape[0], boxes2.shape[0]))
    overlaps = torch.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(boxes1.shape[0]):
        for j in range(boxes2.shape[0]):
            intersection = polygon_intersection(polygons1[i], polygons2[j])
            union = polygon_union(polygons1[i], polygons2[j])
            ious[i, j] = intersection / union
            overlaps[i, j] = intersection

    return overlaps

def boxes_to_polygons(boxes):
    # Implementation to convert boxes to polygons
    polygons = []
    for box in boxes:
        x_min = box[0]
        y_min = box[1]
        x_max = box[2]
        y_max = box[3]
        polygon = shapely.geometry.Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
        polygon = shapely.affinity.rotate(polygon, -1*box[4], use_radians=True)
        polygons.append(polygon)
    return polygons


def polygon_intersection(polygon1, polygon2):
    return shapely.intersection(polygon1, polygon2).area

def polygon_union(polygon1, polygon2):
    # Implementation to calculate union area of polygons
    return shapely.union(polygon1, polygon2).area