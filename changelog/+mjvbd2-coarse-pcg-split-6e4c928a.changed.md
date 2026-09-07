Parallelize sparse products in sufficiently large CUDA MJVBD V2 surface
Galerkin coarse solves while retaining the existing PCG recurrence and
rejection checks. Keep the persistent path for small systems and CPU;
no option migration or change to multilevel enable defaults is required.
