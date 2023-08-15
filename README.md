# MTK

This repository contains the coupled code of [MagmaThermoKinematics.jl](https://github.com/boriskaus/MagmaThermoKinematics.jl) and the pseudo-transient Stokes code of [JustRelax](https://github.com/PTsolvers/JustRelax.jl).
For now, this is a WIP and the code is not yet ready for use. This repository will be used as a testbed for the coupled code and debugging with collaborators.
The implementation of various Julia packages enables the code to run in parallel and with MPI support. 

## TO DO
- add tests for MPI
- stable sticky air setup
- thermal stresses
- add tests for the coupled code
- add particles from [JustPIC](https://github.com/JuliaGeodynamics/JustPIC.jl)
- [...]