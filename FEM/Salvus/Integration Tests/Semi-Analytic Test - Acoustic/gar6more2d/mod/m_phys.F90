!! Copyright INRIA. Contributors : Julien DIAZ and Abdelaaziz EZZIANI
!! 
!! Julien.Diaz@inria.fr and Abdelaaziz.Ezziani@univ-pau.fr
!! 
!! This software is a computer program whose purpose is to
!! compute the analytical solution of problems of waves propagation in two 
!! layered media such as
!! - acoustic/acoustic
!! - acoustic/elastodynamic
!! - acoustic/porous
!! - porous/porous,
!! based on the Cagniard-de Hoop method.
!! 
!! This software is governed by the CeCILL license under French law and
!! abiding by the rules of distribution of free software.  You can  use, 
!! modify and/ or redistribute the software under the terms of the CeCILL
!! license as circulated by CEA, CNRS and INRIA at the following URL
!! "http://www.cecill.info". 
!! 
!! As a counterpart to the access to the source code and  rights to copy,
!! modify and redistribute granted by the license, users are provided only
!! with a limited warranty  and the software's author,  the holder of the
!! economic rights,  and the successive licensors  have only  limited
!! liability. 
!! 
!! In this respect, the user's attention is drawn to the risks associated
!! with loading,  using,  modifying and/or developing or reproducing the
!! software by the user in light of its specific status of free software,
!! that may mean  that it is complicated to manipulate,  and  that  also
!! therefore means  that it is reserved for developers  and  experienced
!! professionals having in-depth computer knowledge. Users are therefore
!! encouraged to load and test the software's suitability as regards their
!! requirements in conditions enabling the security of their systems and/or 
!! data to be ensured and,  more generally, to use and operate it in the 
!! same conditions as regards security. 
!! 
!! The fact that you are presently reading this means that you have had
!! knowledge of the CeCILL license and that you accept its terms.
!! ========================================================================

module m_phys

  implicit none
  integer,save ::  type_medium,type_medium_1,type_medium_2,open
  real*8,save :: mu1, V1, rho1, mu2, V2, rho2,lambda2,VP2,VS2,Vp1
  real*8,save :: rhof1,rhos1,phi1,a1
  real*8,save :: rhof2,rhos2,phi2,a2,vmax
  real*8,save :: rhow2,beta2,M2,la2,R2,ga2,S2,AA2(2,2),B2(2,2)
  real*8,save :: TT2(2,2),P2(2,2),D2(2),Vpsi2,Vf2
  real*8,save :: rhow1,beta1,M1,la1,R1,ga1,S1,AA1(2,2),B1(2,2)
  real*8,save :: TT1(2,2),P1(2,2),D1(2),Vpsi1,Vf1,Vs1,lambda1
  real*8,allocatable,dimension(:,:):: Ux,Uy,P
end module m_phys
