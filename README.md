# Green Lantern: Green's theorem Limb-darkened platform-AgNostic Transiting Ellipsoid Rapid Numerical code

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.14510631-blue?style=for-the-badge)](https://doi.org/10.5281/zenodo.14510630)

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

 - [x] Take time and orbital period as inputs instead of angle $\alpha$
 - [x] Optionally handle elliptical orbits
 - [x] Support `binsize` argument to dual transit flux
 - [x] Support `locked` argument to dual transit flux
 - [ ] Support `eccentric` argument to dual transit flux

## :sparkles: Credits

 + [Funding](https://www.hsfoundation.org/programs/science/51-pegasi-b-fellowship):
   The resources to carry out this project, such as work compensation and
   a custom workstation, were provided by the Heising-Simons Foundation through
   their 51 Pegasi b Fellowship.

<!-- vim: set ft=markdown: -->
