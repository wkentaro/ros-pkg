#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import scipy.ndimage as ndi

from skimage.future.graph import RAG
from skimage.future.graph.rag import _add_edge_filter

from skimage.color import gray2rgb
from skimage.color import label2rgb
from skimage.future.graph import draw_rag
from skimage.measure import regionprops
from skimage.morphology.convex_hull import convex_hull_image
from skimage.util.colormap import viridis


def rag_solidity(labels, connectivity=2):

    graph = RAG()

    # The footprint is constructed in such a way that the first
    # element in the array being passed to _add_edge_filter is
    # the central value.
    fp = ndi.generate_binary_structure(labels.ndim, connectivity)
    for d in range(fp.ndim):
        fp = fp.swapaxes(0, d)
        fp[0, ...] = 0
        fp = fp.swapaxes(0, d)

    # For example
    # if labels.ndim = 2 and connectivity = 1
    # fp = [[0,0,0],
    #       [0,1,1],
    #       [0,1,0]]
    #
    # if labels.ndim = 2 and connectivity = 2
    # fp = [[0,0,0],
    #       [0,1,1],
    #       [0,1,1]]

    ndi.generic_filter(
        labels,
        function=_add_edge_filter,
        footprint=fp,
        mode='nearest',
        output=np.zeros(labels.shape, dtype=np.uint8),
        extra_arguments=(graph,))

    regions = regionprops(labels)
    regions = {r.label: r for r in regionprops(labels)}

    graph.remove_node(0)

    for n in graph:
        region = regions[n]
        graph.node[n].update({'labels': [n],
                              'solidity': region['solidity'],
                              'mask': labels == region.label})

    for x, y, d in graph.edges_iter(data=True):
        new_mask = np.logical_or(graph.node[x]['mask'], graph.node[y]['mask'])
        new_solidity = 1. * new_mask.sum() / convex_hull_image(new_mask).sum()
        org_solidity = np.mean([graph.node[x]['solidity'],
                                graph.node[y]['solidity']])
        d['weight'] = org_solidity / new_solidity

    return graph


if __name__ == '__main__':
    import argparse
    import time
    import matplotlib.pyplot as plt
    from skimage.io import imread
    from util import closed_mask_roi
    from util import masked_slic

    parser = argparse.ArgumentParser()
    parser.add_argument('img_file')
    parser.add_argument('depth_file')
    args = parser.parse_args()

    img_file = args.img_file
    depth_file = args.depth_file

    img = imread(img_file)
    depth = imread(depth_file)
    depth_rgb = gray2rgb(depth)
    plt.figure(figsize=(24, 10))
    plt.subplot(131)
    plt.axis('off')
    plt.imshow(depth_rgb)

    roi = closed_mask_roi(depth)
    roi_labels = masked_slic(img=depth_rgb[roi], mask=depth[roi],
                             n_segments=100, compactness=30)

    labels = np.zeros_like(depth)
    labels[roi] = roi_labels

    out = label2rgb(labels, img, bg_label=0)
    plt.subplot(132)
    plt.axis('off')
    plt.imshow(out)

    g = rag_solidity(labels)

    out = draw_rag(labels, g, img, colormap=viridis, desaturate=True)
    plt.subplot(133)
    plt.axis('off')
    plt.imshow(out)

    # plt.show()
    plt.savefig('{}_rag_solidity.png'.format(time.time()))
