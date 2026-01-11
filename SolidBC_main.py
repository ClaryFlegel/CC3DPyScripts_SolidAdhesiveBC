import cc3d
import sys


from cc3d.core.PySteppables import SteppableBasePy

from cc3d.CompuCellSetup.CC3DCaller import CC3DSimService
from cc3d.core.PyCoreSpecs import PottsCore, CellTypePlugin, VolumePlugin, ContactPlugin, CenterOfMassPlugin, PixelTrackerPlugin, BoundaryPixelTrackerPlugin, NeighborTrackerPlugin, ExternalPotentialPlugin, UniformInitializer

from Steppable_S import WoundMakerSteppable
from Parameters import *
from Measurements import Measurements
from CellVolumeMeasurements import CellVolumeMeasurement

cc3d.CC3D_OpenCL_enabled = False
run_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

def specs_gen():

    specs = [PottsCore(dim_x=grid_x, dim_y=grid_y, neighbor_order=1)]

    cell_types = CellTypePlugin("Cell","Wall","Fluid")
    cell_types.frozen_set("Wall", True)
    specs.append(cell_types)


    volume_specs = VolumePlugin()
    volume_specs.param_new("Cell", target_volume=target_volume, lambda_volume=lambda_volume)
    volume_specs.param_new("Wall", target_volume=1, lambda_volume=1)
    volume_specs.param_new("Fluid", target_volume=1, lambda_volume=0)
    specs.append(volume_specs)



    contact_specs = ContactPlugin(neighbor_order=4)
    contact_specs.param_new(type_1="Medium", type_2="Cell", energy=10)
    contact_specs.param_new(type_1="Cell", type_2="Cell", energy=10)
    contact_specs.param_new(type_1="Medium", type_2="Wall", energy=10)
    contact_specs.param_new(type_1="Wall", type_2="Wall", energy=10)
    contact_specs.param_new(type_1="Medium", type_2="Fluid", energy=10)
    contact_specs.param_new(type_1="Fluid", type_2="Fluid", energy=10)

    contact_specs.param_new(type_1="Cell", type_2="Wall", energy=100)
    contact_specs.param_new(type_1="Cell", type_2="Fluid", energy=10)
    contact_specs.param_new(type_1="Wall", type_2="Fluid", energy=10)

    specs.append(contact_specs)


    specs.append(PixelTrackerPlugin())
    specs.append(BoundaryPixelTrackerPlugin())

    specs.append(ExternalPotentialPlugin(com_based=True))

    specs.append(CenterOfMassPlugin())
    specs.append(NeighborTrackerPlugin())
    


    unif_init_specs = UniformInitializer()
    unif_init_specs.region_new(pt_min=(0,0,0), pt_max=(grid_x,grid_y,1), gap=0, width=10, cell_types=["Cell"])
    #unif_init_specs.region_new(pt_min=(1,1,0), pt_max=(grid_x-1,grid_y-1,1), gap=0, width=1, cell_types= ["Fluid"])
    #unif_init_specs.region_new(pt_min=(3,3,0), pt_max=(grid_x-3,grid_y-3,1), gap=0, width=10, cell_types=["Cell"])
    specs.append(unif_init_specs)

    return specs 
    

import random
import numpy as np 




#from pathlib import Path

if __name__ == '__main__':


# create file path
    #output_file = run_dir / f"simulation_results_{run_id}.txt"

    #output_file = f"simulation_results_{run_id}.txt"

    
    # clear old results
    #with open(output_file, "w") as f:
    #    f.write("CC3D Simulation Results\n")
    
    random.seed(run_id)
    np.random.seed(run_id)

    specs=specs_gen()
    sim = CC3DSimService()
    sim.register_specs(specs)
    sim.register_steppable(steppable=WoundMakerSteppable(frequency=1,run_id=run_id))
    measurements_steppable = Measurements(frequency=1,run_id=run_id)
    sim.register_steppable(steppable=measurements_steppable)
    sim.register_steppable(steppable=CellVolumeMeasurement(frequency=1, run_id=run_id))
    sim.run()
    sim.init()
    sim.start()

    sim.visualize()


    #input('Press any key to continue...')


    while sim.current_step < t and not measurements_steppable.wound_closed_flag:
        sim.step()
    #with open(output_file, "a") as f:
    #    f.write(sim.profiler_report + "\n")



    # cleanup: if zsh segmentation fault due to closing issue
    #del sim
    #import gc
    #gc.collect()
  

    
    
    

    

    