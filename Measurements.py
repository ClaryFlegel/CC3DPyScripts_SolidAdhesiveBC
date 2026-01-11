import cc3d
from cc3d.core.PySteppables import SteppableBasePy
import numpy as np 
import math

from Parameters import *
from pathlib import Path

class Measurements(SteppableBasePy):
    def __init__(self, frequency=1, run_id=0):
        super().__init__(frequency=frequency)

        self.run_id=run_id
        self.wound_closed_flag = False # it is not yet opened really

    def start(self,run_id=0):
        self.run_dir = Path("SolidRuns") / f"Lx{grid_x}_Ly{grid_y}" / f"R{wR}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Optional: delete all old measurement files for this wound folder
        #for f in self.run_dir.glob("simulation_results_*.txt"):
        #    f.unlink()
            #print(f"[Measurements] Deleted old file {f.name}")

        # create file path
        self.output_file = self.run_dir / f"simulation_results_{self.run_id}.txt"

        #self.output_file = f"simulation_results_{run_id}.txt"
        #with open(self.output_file, "a") as f:
        #    f.write("mcs, wound Area\n")

        with open(self.output_file, "w") as f:
            f.write(f"# Domain Size: Lx={grid_x}, Ly={grid_y}\n")
            f.write(f"# Wound Radius Created: R={wR}\n")
            f.write("mcs,woundArea\n")
  
    def step(self,mcs):
        #print(f"Measurements step called at MCS={mcs}")  # Debug line
        # woundArea=0
        # occupiedArea=0
        # for cell in self.cell_list:
        #     occupiedArea += cell.volume
        # woundArea=(grid_x-3)*(grid_y-3) - occupiedArea
        woundArea = self.compute_wound_area()
        
        with open(self.output_file, "a") as f:
            f.write(f"{mcs},{woundArea}\n")
        
        if mcs > woundMakerTime and not self.wound_closed_flag:
            if woundArea == 0:
                self.closed_counter += 1
            else:
                self.closed_counter = 0

            if self.closed_counter >= 3:
                print(f"Wound stably closed at MCS {mcs}")
                self.wound_closed_flag = True
                self.stop_simulation()


    def compute_wound_area(self):
        woundArea = 0
        for x in range(grid_x):
            for y in range(grid_y):
                if self.cellField[x, y, 0] is None:  # medium pixel
                    woundArea += 1
        return woundArea