# Standard Python packages
import toml
import math
from pathlib import Path
import json
import h5py
import os

# Workflow management.
from salvus.flow import api
import salvus.toolbox.toolbox as st

# Specific objects to aid in setting up simulations.
import salvus.flow.simple_config as sc

SALVUS_FLOW_SITE_NAME = os.environ.get("SITE_NAME", "tutorials")
ranks = 2

import sys
sys.path.insert(0,'/home/nolsen/Tutorials/PDM/Helpers')

from Helpers import *
from MyCrack import *

# Opening the config file 

config_file_path = sys.argv[1]

experiment_name = config_file_path.split('/')[1].split('.')[0]

f = open(config_file_path) 
inputs = json.load(f) 
f.close()

# Extracting the parameters
nelem_x = inputs['mesh']['nelem_x']
nelem_y = inputs['mesh']['nelem_y']

min_x = inputs['mesh']['x_min']
max_x = inputs['mesh']['x_max']

min_y = inputs['mesh']['y_min']
max_y = inputs['mesh']['y_max']

solid_vp = inputs['materials'][0]['parameters']['VP']
solid_rho = inputs['materials'][0]['parameters']['RHO']

# Construct the mesh
mesh = get_mesh(nelem_x=nelem_x,nelem_y=nelem_y,max_x=max_x,max_y=max_y)

a = inputs['fracture'][0]['parameters']['a']
b = inputs['fracture'][0]['parameters']['b']
x_center = inputs['fracture'][0]['parameters']['x_center']
y_center = inputs['fracture'][0]['parameters']['y_center']

fracture_obj = MyRectangularCrack(nelem_x=nelem_x,nelem_y=nelem_y,max_x=max_x,max_y=max_y,a=a,b=b,x_center=x_center,y_center=y_center)

linear_fracture = fracture_obj.getLinearCrack()

# a mask for the solid elements
n_linear_fracture = np.logical_not(linear_fracture)

# attach material properties
vpa = 0
rhoa = 0

for i, mat in enumerate(inputs['materials']):
    vpa += n_linear_fracture * mat['parameters']['VP'] if mat['fracture'] else linear_fracture * mat['parameters']['VP']
    rhoa += n_linear_fracture * mat['parameters']['RHO'] if mat['fracture'] else linear_fracture * mat['parameters']['RHO']

vp_vec = vpa.reshape(mesh.nelem,1).repeat(mesh.nodes_per_element,axis=1)
rho_vec = rhoa.reshape(mesh.nelem,1).repeat(mesh.nodes_per_element,axis=1)

for name, material in zip(['VP','RHO'],[vp_vec,rho_vec]):
    mesh.attach_field(name, material)

mesh.attach_field("fluid", np.ones(mesh.nelem)) # Ones only for the acoustic

# # # Sources
srcs = [
    sc.source.cartesian.ScalarPoint2D(
            x=src['location']['x'],
            y=src['location']['y'],
            f=src['location']['fx'], 
            source_time_function=sc.stf.Ricker(center_frequency=src['frequency']))
         for _i, src in enumerate(inputs['sources'])   
]

# # # Receivers
recs = [
    sc.receiver.cartesian.Point2D(
        x=rec['location']['x'], 
        y=rec['location']['y'], 
        station_code=f"{_i:03d}", 
        fields=[rec['field']])
    for _i, rec in enumerate(inputs['receivers'])
]

output_path = inputs['simulation']['output_path'] + "/" + experiment_name

for _i, src in enumerate(srcs):
    sim = sc.simulation.Waveform(
        mesh=mesh, sources=src, receivers=recs
    )

    print("\nEVENT{0:04}".format(1))
    plot_setup(0.25,[src],recs)

    sim.output.point_data.format = "hdf5"
    sim.physics.wave_equation.start_time_in_seconds = inputs['simulation']['start_time']
    sim.physics.wave_equation.end_time_in_seconds = inputs['simulation']['end_time']
    sim.physics.wave_equation.time_step_in_seconds = inputs['simulation']['end_time'] / 10000

    sim.output.volume_data.format = "hdf5"
    sim.output.volume_data.fields = ["phi"]
    sim.output.volume_data.filename = "output.h5"
    sim.output.volume_data.sampling_interval_in_time_steps = 10
    
    output_folder = Path(output_path + "/EVENT{0:04}".format(_i))

    # Run.
    api.run(
        input_file=sim,
        site_name=SALVUS_FLOW_SITE_NAME,
        ranks=ranks,
        output_folder=output_folder,
        overwrite=True,  
        get_all=True   
    )