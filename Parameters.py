
import cc3d
from cc3d.core.PySteppables import SteppableBasePy
#from cc3d.core.PyCoreSpecs import PixelTrackerPlugin, BoundaryPixelTrackerPlugin, NeighborTrackerPlugin
import numpy as np 
import math

#important! need square domain from wound creation logic
grid_x = 256 #206
grid_y = grid_x


woundMakerTime=10 #mcs when wound is created
wR = 60 # wound radius in pixels 
target_volume, lambda_volume = 100, 0.5

N=30 #repeated runs
t=100001 #not inclusive: last mcs = t-1 ----- Maximum MCS 
