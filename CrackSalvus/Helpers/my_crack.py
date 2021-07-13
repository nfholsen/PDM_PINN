import numpy as np
import os 

from salvus.flow import api
import salvus.flow.simple_config as sc
from salvus.mesh.simple_mesh import rho_from_gardeners


import h5py

import sys
sys.path.append('/home/nolsen/Salvus/Salvus_PDM/CrackSalvus/Helpers/')

from helpers import *


SALVUS_FLOW_SITE_NAME = os.environ.get("SITE_NAME", "local")
ranks = 4

class MyCrack:
    def __init__(self,nelem_x,nelem_y,max_x,max_y):
        self.nelem_x = nelem_x
        self.nelem_y = nelem_y
        self.max_x = max_x
        self.max_y = max_y

class MyEllipticalCrack(MyCrack):
    def __init__(self,nelem_x,nelem_y,max_x,max_y,a,b,x_center,y_center):
        MyCrack.__init__(self,nelem_x,nelem_y,max_x,max_y)
        self.a = a
        self.b = b
        self.x_center = x_center
        self.y_center = y_center

    def getCrack(self):

        x = np.linspace(0, self.max_x, self.nelem_x)
        y = np.linspace(self.max_y, 0, self.nelem_y)
        
        yy, xx = np.meshgrid(y, x, indexing="ij")

        crack = np.zeros((self.nelem_x,self.nelem_y),dtype="bool")

        mask = ((self.x_center - xx) ** 2)/(self.a ** 2) + ((self.y_center - yy) ** 2)/(self.b ** 2) <= 1

        crack[mask] = True
        return crack

    def getLinearCrack(self):
        crack = self.getCrack()
        return crack.ravel()


class MySphericalCrack(MyCrack):
    def __init__(self,nelem_x,nelem_y,max_x,max_y,r,x_center,y_center):
        MyCrack.__init__(self,nelem_x,nelem_y,max_x,max_y)
        self.r = r
        self.x_center = x_center
        self.y_center = y_center

    def getCrack(self):
        
        x = np.linspace(0, self.max_x, self.nelem_x)
        y = np.linspace(self.max_y, 0, self.nelem_y)
        
        yy, xx = np.meshgrid(y, x, indexing="ij")

        crack = np.zeros((self.nelem_x,self.nelem_y),dtype="bool")

        mask = ((xx - self.x_center) ** 2) + ((yy - self.y_center) ** 2) <= self.r ** 2

        crack[mask] = True
        return crack

    def getLinearCrack(self):
        crack = self.getCrack()
        return crack.ravel()

class MyRectangularCrack(MyCrack):
    def __init__(self,nelem_x,nelem_y,max_x,max_y,a,b,x_center,y_center):
        MyCrack.__init__(self,nelem_x,nelem_y,max_x,max_y)
        self.a = a
        self.b = b
        self.x_center = x_center
        self.y_center = y_center

    def getCrack(self):
        
        x = np.linspace(0, self.max_x, self.nelem_x)
        y = np.linspace(self.max_y, 0, self.nelem_y)
        
        yy, xx = np.meshgrid(y, x, indexing="ij")

        crack = np.zeros((self.nelem_x,self.nelem_y),dtype="bool")

        mask = (xx < self.x_center + self.a/2) & (xx > self.x_center - self.a/2) & (yy < self.y_center + self.b/2) & (yy > self.y_center - self.b/2)

        crack[mask] = True
        return crack

    def getLinearCrack(self):
        crack = self.getCrack()
        return crack.ravel()

################################################
# # # Class for salvus ACOUSTIC simulation # # #
################################################

class SalvusSimulation():
    def __init__(self, mesh, srcs, recs, output_folder):
        # Elements of the simulation
        self.mesh = mesh 
        self.srcs = srcs
        self.recs = recs
        self.output_folder = output_folder

    def run(self,start_time,end_time,time_step,wavefield_sampling=10):

        for _i, src in enumerate(self.srcs):
            sim = sc.simulation.Waveform(
                mesh=self.mesh, 
                sources=src, 
                receivers=self.recs
            )

            print("\nEVENT{0:04}".format(_i))

            plot_setup(0.25,[src],self.recs)

            sim.output.point_data.format = "hdf5"
            sim.physics.wave_equation.start_time_in_seconds = start_time
            sim.physics.wave_equation.end_time_in_seconds = end_time
            sim.physics.wave_equation.time_step_in_seconds = time_step

            sim.output.volume_data.format = "hdf5"
            sim.output.volume_data.fields = ["phi"]
            sim.output.volume_data.filename = "output.h5"
            sim.output.volume_data.sampling_interval_in_time_steps = wavefield_sampling

            simulation_number = "/EVENT{0:04}".format(_i)

            # Run.
            api.run(
                input_file=sim,
                site_name=SALVUS_FLOW_SITE_NAME,
                ranks=ranks,
                output_folder=self.output_folder + simulation_number,
                overwrite=True,  
                get_all=True   
            )

######################
# # # Class Mesh # # #
######################

class Mesh():
    def __init__(self, mesh, homogeneous_material):

        self.mesh = mesh

        # Define base materials
        self.vpa = 0
        self.rhoa = 0

        self.vpa += np.ones(self.mesh.nelem) * homogeneous_material['parameters']['VP']
        self.rhoa += np.ones(self.mesh.nelem) * homogeneous_material['parameters']['RHO']

    def attach_materials(self):

        self.vp_vec = self.vpa.reshape(self.mesh.nelem,1).repeat(self.mesh.nodes_per_element,axis=1)
        self.rho_vec = self.rhoa.reshape(self.mesh.nelem,1).repeat(self.mesh.nodes_per_element,axis=1)

        for name, material in zip(['VP','RHO'],[self.vp_vec,self.rho_vec]):
            self.mesh.attach_field(name, material)

        self.mesh.attach_field("fluid", np.ones(self.mesh.nelem)) # Ones only for the acoustic

        return self.mesh

    def add_crack(self, fracture_obj, materials):

        linear_fracture = fracture_obj.getLinearCrack()

        self.vpa, self.rhoa = np.zeros((2, self.mesh.nelem))

        # a mask for the solid elements
        n_linear_fracture = np.logical_not(linear_fracture)
        
        for i, mat in enumerate(materials):
            self.vpa += n_linear_fracture * mat['parameters']['VP'] if mat['name'] == 'fracture' else linear_fracture * mat['parameters']['VP']
            self.rhoa += n_linear_fracture * mat['parameters']['RHO'] if mat['name'] == 'fracture' else linear_fracture * mat['parameters']['RHO']

        self.mesh = self.attach_materials()

        return self.mesh, {'VP':self.vpa,'RHO':self.rhoa,'Fracture':linear_fracture}

    def add_layers(self,vp_index,vp,rho):

        self.vpa, self.rhoa = np.ones((2, self.mesh.nelem))

        for _i, index in enumerate(vp_index):

            # Find which elements are in a given region.
            idx = np.where(self.mesh.elemental_fields["region"] == _i)

            # Set parameters in that region to a constant value.
            self.vpa[idx] = vp[index]
            self.rhoa[idx] = rho[index]