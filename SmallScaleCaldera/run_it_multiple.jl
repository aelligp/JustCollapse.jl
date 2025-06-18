using DataFrames, CSV, Dates

function main()
    Systematics = true
    results = DataFrame(conduit=Float64[], depth=Float64[], radius=Float64[], ar=Float64[], extension=Float64[], diameter=Float64[], friction_angle=Float64[])

    for systematic in Systematics

        conduits = 1.5e-1
        depths = 5e0
        radii = 1.5#:0.25:2.5e0
        ars = 1.0#:0.5:2.5e0
        extensions = 1e-15 #, 5e-15, 1e-14, 5e-14, 1e-13
        friction = 15:5:30.0
        max_jobs = 4
        job_counter = 0
        batch_counter = 0

        for conduit in conduits, depth in depths, radius in radii, ar in ars, extension in extensions, fric_angle in friction
            diameter = 2 * (radius * ar)
            if diameter <= 8.0
                if job_counter == 0
                    # Create a new batch file
                    batch_counter += 1
                    open("runme_batch_$(batch_counter).sh", "w") do io
                        println(io, """#!/bin/bash -l
#SBATCH --job-name="Batch_$(batch_counter)_$(today())"
#SBATCH --output=batch_$(batch_counter).o
#SBATCH --error=batch_$(batch_counter).e
#SBATCH --time=00:30:00 #HH:MM:SS
#SBATCH --nodes=1
#SBATCH --ntasks=$(max_jobs)
#SBATCH --gpus-per-node=1
#SBATCH --constraint=gpu
#SBATCH --account=c44

export MPICH_GPU_SUPPORT_ENABLED=1
export IGG_CUDAAWARE_MPI=1 # IGG
export JULIA_CUDA_USE_COMPAT=false # IGG

export LD_PRELOAD=/capstor/scratch/cscs/paellig/.julia/gh200/juliaup/depot/artifacts/4c845485a01f4a5c51481d9714303da1309c266f/lib/libcrypto.so.3

""")
                    end
                end

                # Append the `srun` call to the current batch file
                open("runme_batch_$(batch_counter).sh", "a") do io
                    if job_counter == max_jobs-1  # Last job in the batch
                        println(io, """
srun --exclusive -N1 -n1 --gpu-bind=per_task:1 --cpu_bind=sockets julia --project -t 12 SmallScaleCaldera/Caldera2D.jl $(conduit) $(depth) $(radius) $(ar) $(extension) $(fric_angle) /capstor/scratch/cscs/paellig/jobreport print report""")
                    else
                        println(io, """
srun --exclusive -N1 -n1 --gpu-bind=per_task:1 --cpu_bind=sockets julia --project -t 12 SmallScaleCaldera/Caldera2D.jl $(conduit) $(depth) $(radius) $(ar) $(extension) $(fric_angle) /capstor/scratch/cscs/paellig/jobreport print report &""")
                    end
                end

                job_counter += 1
                push!(results, (conduit, depth, radius, ar, extension, diameter, fric_angle))

                if job_counter == max_jobs
                    job_counter = 0
                end
            else
                println("Diameter too large: $(diameter)")  # Debugging output
            end
        end

        # Submit all batch files if any were created
        if batch_counter > 0
            for i in 1:batch_counter
                # run(`sbatch runme_batch_$(i).sh`)
                println("Batch file runme_batch_$(i).sh submitted")
            end

            # Remove all batch files after submission
            sleep(1)
            for i in 1:batch_counter
                # rm("runme_batch_$(i).sh")
                println("Batch file runme_batch_$(i).sh removed")
            end
        end
    end

    # Write DataFrame to CSV
    CSV.write(joinpath(@__DIR__, "Systematics_$(today()).csv"), results)
end

main()
