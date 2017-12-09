from skimage.future import graph
from skimage import data, segmentation, color, filters, io
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

input_filename="azeer.jpg"
im = Image.open(input_filename)
img = np.array(im)

#img = data.coffee()
gimg = color.rgb2gray(img)

labels = segmentation.slic(img, compactness=30, n_segments=2000)
edges = filters.sobel(gimg)
edges_rgb = color.gray2rgb(edges)

g = graph.rag_boundary(labels, edges)
lc = graph.show_rag(labels, g, edges_rgb, img_cmap=None, edge_cmap='viridis',
                    edge_width=1.2)

plt.colorbar(lc, fraction=0.03)
io.show()