using DataFrames, CSV, Dates

function main()
    Systematics = true
    results = DataFrame(depth=Float64[], radius=Float64[], ar=Float64[], extension=Float64[], diameter=Float64[], friction_angle=Float64[])

    for systematic in Systematics

        depths = 5e0
        radii = 1.5:0.25:2.0#2.5e0
        ars = 1.0:0.5:2.5e0
        extensions = 1e-15 #, 5e-15, 1e-14, 5e-14, 1e-13
        friction = 15:5:30.0
        for depth in depths, radius in radii, ar in ars, extension in extensions, fric_angle in friction
            jobname = "Caldera_$(today())_$(depth)_$(radius)_$(ar)_$(extension)_$(fric_angle)"
            diameter = 2*(radius*ar)
            str =
"""#!/bin/bash -l
#SBATCH --job-name=\"$(jobname)\"
#SBATCH --output=caldera_$(jobname).o
#SBATCH --error=caldera_$(jobname).e
#SBATCH --time=24:00:00 #HH:MM:SS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=gpu
#SBATCH --account=sm109

export MPICH_GPU_SUPPORT_ENABLED=1
export IGG_CUDAAWARE_MPI=1 # IGG
export JULIA_CUDA_USE_COMPAT=false # IGG
export LD_PRELOAD=export LD_PRELOAD=/capstor/scratch/cscs/paellig/.julia/gh200/juliaup/depot/artifacts/e313763c2521efed920ca04560b24511b37d35ea/lib/libcrypto.so.3

# mount the uenv prgenv-gnu with the view named default
srun --constraint=gpu --gpu-bind=per_task:1 --cpu_bind=sockets -- julia --project -t 12 SmallScaleCaldera/Caldera2D.jl $(depth) $(radius) $(ar) $(extension) $(fric_angle) """
            if diameter <= 8.0
                open("runme_test.sh", "w") do io
                    println(io, str)
                end

                # Submit the job
                run(`sbatch runme_test.sh`)
                println("Job submitted")
                # remove the file
                sleep(1)
                rm("runme_test.sh")
                println("File removed")
                # Append parameters to DataFrame
                push!(results, (depth, radius, ar, extension, diameter, fric_angle))
            else
                println("Diameter too large")
            end
        end
    end

    # Write DataFrame to CSV
    CSV.write(joinpath(@__DIR__, "Systematics_$(today()).csv"), results)
end

main()
