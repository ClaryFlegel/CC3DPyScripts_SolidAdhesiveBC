import cc3d
from cc3d.core.PySteppables import SteppableBasePy
#from cc3d.core.PyCoreSpecs import PixelTrackerPlugin, BoundaryPixelTrackerPlugin, NeighborTrackerPlugin
import numpy as np 
import math

from Parameters import *



class WoundMakerSteppable(SteppableBasePy):
    def __init__(self, frequency=1,run_id=0):
        super().__init__(frequency=frequency)
        self.run_id = run_id
        self.wound_made = False   

    def start(self):
        # safe: start() is called once per simulation
        print(f"Initializing wound for run {self.run_id}")


    def step(self,mcs):
        #print(f"2: Measurements step called at MCS={mcs}")  # Debug line
        min_vol = 1.0
        for cell in self.cell_list_by_type(self.CELL):
            cell.lambdaVolume = lambda_volume*(cell.volume + target_volume)/(max(cell.volume,min_vol))

        left=int(self.dim.x/2 - wR)
        right=int(self.dim.x/2 + wR)
        top=int(self.dim.y/2 - wR)
        bottom=int(self.dim.y/2 + wR)
        counter=0
        #cells_to_delete = set()

        if not self.wound_made and mcs == woundMakerTime:  # pick an MCS after cells appear  
            for x in range(left, right):
                for y in range(top, bottom):
                    cell = self.cellField[x, y, 0]
                    if cell:
                        self.deleteCell(cell)
                        #cells_to_delete.add(cell)
                        counter+=1
            #for cell in cells_to_delete:
                #self.deleteCell(cell)

            self.wound_made = True
            print("Wound created at MCS =", mcs)
            print("Wound size in cells =",counter)

        if mcs > woundMakerTime: 
            for cell in self.cell_list_by_type(self.CELL):
                if cell is None:
                    print("cell_list_by_type returned None!")
                vector = self.get_local_polarity_vector(cell)
                force = 1200
                # Make sure ExternalPotential plugin is loaded
                cell.lambdaVecX = -force*vector[0]  # force component pointing along X axis - towards positive X's
                cell.lambdaVecY = -force*vector[1]  # force component pointing along Y axis - towards negative Y's
                # cell.lambdaVecZ = 0.0  force component pointing along Z axis
        

    def get_local_polarity_vector(self, cell):
        boundary_pixels = self.get_cell_boundary_pixel_list(cell)

        pixels = np.array([[p.pixel.x,p.pixel.y] for p in boundary_pixels])
        vec = pixels - np.array([cell.xCOM,cell.yCOM])
        norm = np.linalg.norm(vec,axis=1,keepdims=True)
        vec_norm = np.divide(vec,norm,where=norm>1e-6)
        pixels_proj = np.round(pixels + vec_norm).astype(int)
        mask_inside = (
            (pixels_proj[:,0] >= 0) &
            (pixels_proj[:,0] < self.dim.x) &
            (pixels_proj[:,1] >= 0) &
            (pixels_proj[:,1] < self.dim.y)
        )
        pixels_proj = pixels_proj[mask_inside]

        medium_check = [False if self.cellField[int(p[0]), int(p[1]), 0] else True for p in pixels_proj]

        if np.any(medium_check): #any entry is 1 (true) 
            vec_norm=vec_norm[mask_inside][medium_check]
            vector = np.average(vec_norm, axis=0)
            return vector
            
        else:
            vector = np.array([0.0, 0.0])
            return vector
            
        

    