# Green Lantern: Green's theorem Limb-darkened platform-AgNostic Transiting Ellipsoid Rapid Numerical code

![Project logo](/assets/logo.png)

This code was developed by Ellen M. Price ([@emprice](https://github.com/emprice))
while working as a 51 Pegasi b Fellow, sponsored by the Heising-Simons Foundation,
at the University of Chicago in Chicago, IL, USA.

 + [External dependencies](#cactus-external-dependencies)
 + [To-do list](#white_check_mark-to-do-list)
 + [Credits](#sparkles-credits)

## :cactus: External dependencies

 + [OpenCL](https://www.khronos.org/opencl): OpenCL is an open and standard
   language for programming on many kinds of compute devices, particularly
   GPUs. It is supported by AMD, Intel, and NVIDIA, meaning that only one
   version of a compute kernel need be written for a project, and it can
   be compiled to run on any device that supports the standard. To build
   this code, the OpenCL headers and the implementation library should
   be available.
 + [pocky](https://github.com/emprice/pocky): Pocky is a simple Python
   bridge to OpenCL that provides a consistent API for accessing device
   memory and queues. It can be installed manually from GitHub using pip.

## :white_check_mark: To-do list

 - [ ] Take time and orbital period as inputs instead of angle $\alpha$
 - [ ] Optionally handle elliptical orbits

## :sparkles: Credits

 + [Funding](https://www.hsfoundation.org/programs/science/51-pegasi-b-fellowship):
   The resources to carry out this project, such as work compensation and
   a custom workstation, were provided by the Heising-Simons Foundation through
   their 51 Pegasi b Fellowship.

<!-- vim: set ft=markdown: -->
