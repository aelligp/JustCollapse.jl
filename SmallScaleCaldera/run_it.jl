using DataFrames, CSV, Dates

function main()
    Systematics = true
    results = DataFrame(conduit=Float64[], depth=Float64[], radius=Float64[], ar=Float64[], extension=Float64[], diameter=Float64[])

    for systematic in Systematics

        conduits = 1.5e-1
        depths = 5e0
        radii = 1.5:0.25:2.5e0
        ars = 1.0:0.5:2.5e0
        extensions = 1e-15 #, 5e-15, 1e-14, 5e-14, 1e-13
        for conduit in conduits, depth in depths, radius in radii, ar in ars, extension in extensions
            jobname = "NO_Ext_Systematics_$(today())_$(conduit)_$(depth)_$(radius)_$(ar)_$(extension)"
            diameter = 2*(radius*ar)
            str =
"""#!/bin/bash -l
#SBATCH --job-name=\"$(jobname)\"
#SBATCH --output=caldera_$(jobname).o
#SBATCH --error=caldera_$(jobname).e
#SBATCH --time=21:00:00 #HH:MM:SS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=gpu
#SBATCH --account=c44

export MPICH_GPU_SUPPORT_ENABLED=1
export IGG_CUDAAWARE_MPI=1 # IGG
export JULIA_CUDA_USE_COMPAT=false # IGG

# mount the uenv prgenv-gnu with the view named default
srun --constraint=gpu --gpu-bind=per_task:1 --cpu_bind=sockets /capstor/scratch/cscs/paellig/jobreport -o report -- julia --project -t 12 SmallScaleCaldera/Caldera2D.jl $(conduit) $(depth) $(radius) $(ar) $(extension) /capstor/scratch/cscs/paellig/jobreport print report"""
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
                push!(results, (conduit, depth, radius, ar, extension, diameter))
            else
                println("Diameter too large")
            end
        end
    end

    # Write DataFrame to CSV
    CSV.write(joinpath(@__DIR__, "Systematics_$(today()).csv"), results)
end

main()
