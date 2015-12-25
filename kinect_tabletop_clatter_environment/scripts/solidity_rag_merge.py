#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

import numpy as np
from skimage.color import gray2rgb
from skimage.color import label2rgb
from skimage.future import graph
from skimage.morphology.convex_hull import convex_hull_image

from rag_solidity import rag_solidity


def _solidity_weight_func(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing solidity."""
    org_solidity = np.mean([graph.node[src]['solidity'],
                            graph.node[dst]['solidity']])
    new_mask1 = np.logical_or(graph.node[src]['mask'], graph.node[n]['mask'])
    new_mask2 = np.logical_or(graph.node[dst]['mask'], graph.node[n]['mask'])
    new_solidity1 = 1. * new_mask1.sum() / convex_hull_image(new_mask1).sum()
    new_solidity2 = 1. * new_mask2.sum() / convex_hull_image(new_mask2).sum()
    weight1 = org_solidity / new_solidity1
    weight2 = org_solidity / new_solidity2
    return min([weight1, weight2])


def _solidity_merge_func(graph, src, dst):
    """Callback called before merging two nodes of a solidity graph."""
    new_mask = np.logical_or(graph.node[src]['mask'], graph.node[dst]['mask'])
    graph.node[dst]['mask'] = new_mask
    graph.node[dst]['solidity'] = \
        1. * np.sum(new_mask) / np.sum(convex_hull_image(new_mask))


def main():
    import argparse
    import matplotlib
    import matplotlib.pyplot as plt
    import os
    import time
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
    plt.subplot(231)
    plt.axis('off')
    plt.imshow(depth_rgb)

    t_start = time.time()
    roi = closed_mask_roi(depth)
    roi_labels = masked_slic(img=depth_rgb[roi], mask=depth[roi],
                             n_segments=20, compactness=30)
    print('slic: takes {} [secs]'.format(time.time() - t_start))

    labels = np.zeros_like(depth)
    labels[roi] = roi_labels
    print('n_labels: {}'.format(len(np.unique(labels))))

    out = label2rgb(labels, img, bg_label=0)
    plt.subplot(232)
    plt.axis('off')
    plt.imshow(out)

    t_start = time.time()
    g = rag_solidity(labels, connectivity=2)
    print('rag_solidity: takes {} [secs]'.format(time.time() - t_start))

    # draw rag: all
    out = graph.draw_rag(labels, g, img)
    plt.subplot(233)
    plt.axis('off')
    plt.imshow(out)
    # draw rag: good edges
    cmap = matplotlib.colors.ListedColormap(['#0000FF', '#FF0000'])
    out = graph.draw_rag(labels, g, img, node_color='#00FF00', colormap=cmap,
                         thresh=1, desaturate=True)
    plt.subplot(234)
    plt.axis('off')
    plt.imshow(out)

    t_start = time.time()
    merged_labels = graph.merge_hierarchical(
        labels, g, thresh=1, rag_copy=False,
        in_place_merge=True,
        merge_func=_solidity_merge_func, weight_func=_solidity_weight_func)
    print('graph.merge_hierarchical: takes {} [secs]'.format(time.time() - t_start))

    out = label2rgb(merged_labels, img, bg_label=0)
    plt.subplot(235)
    plt.axis('off')
    plt.imshow(out)

    merged_g = rag_solidity(merged_labels)
    # draw rag: all
    out = graph.draw_rag(merged_labels, merged_g, img)
    plt.subplot(236)
    plt.axis('off')
    plt.imshow(out)

    # plt.show()
    plt.savefig('{}_solidity_rag_merge.png'.format(time.time()), dpi=100)


if __name__ == '__main__':
    main()
