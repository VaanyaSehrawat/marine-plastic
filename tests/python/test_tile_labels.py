import numpy as np

from ml.data import tile_labels as tl


def test_tile_boxes_cover_image_with_full_size_tiles():
    boxes = tl.tile_boxes(100, 130, tile=50)
    assert all((x1 - x0, y1 - y0) == (50, 50) for x0, y0, x1, y1 in boxes)
    assert max(b[2] for b in boxes) == 130 and max(b[3] for b in boxes) == 100


def test_labels_positive_negative_uncertain():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:40, 10:40] = True                      # big object in the top-left tile
    mask[10:12, 55:57] = True                      # tiny sliver in the top-right tile
    boxes = tl.tile_boxes(100, 100, tile=50)       # tiles: TL, TR, BL, BR
    labels, fracs = tl.label_tiles(mask, boxes, pos_frac=0.02, margin=4)
    by = dict(zip(boxes, labels.tolist()))
    assert by[(0, 0, 50, 50)] == tl.POS
    assert by[(50, 0, 100, 50)] == tl.UNC          # sliver is below pos_frac but not "clean"
    assert by[(0, 50, 50, 100)] == tl.NEG and by[(50, 50, 100, 100)] == tl.NEG


def test_tile_next_to_object_is_uncertain_not_negative():
    mask = np.zeros((100, 100), dtype=bool)
    mask[0:50, 40:49] = True                       # object just left of the right tile boundary at x=50
    boxes = [(50, 0, 100, 50)]
    labels, _ = tl.label_tiles(mask, boxes, pos_frac=0.02, margin=16)
    assert labels[0] == tl.UNC


def test_union_mask_polygon_and_bbox_fallback():
    anns = [{"segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]]}, {"segmentation": [], "bbox": [50, 50, 10, 10]}]
    m = tl.union_mask(anns, 100, 100)
    assert m[20, 20] and m[55, 55] and not m[80, 80]
