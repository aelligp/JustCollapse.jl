using DataFrames, CSV, Dates

function main()
    Systematics = true
    results = DataFrame(conduit=Float64[], depth=Float64[], radius=Float64[], ar=Float64[], extension=Float64[], diameter=Float64[])

    for systematic in Systematics

        conduits = 1.5e-1
        depths = 3e0:0.5:5e0
        radii = 1e0:0.5:2.5e0
        ars = 1e0:0.5:2e0
        extensions = 1e-15 #, 5e-15, 1e-14, 5e-14, 1e-13
        for conduit in conduits, depth in depths, radius in radii, ar in ars, extension in extensions
            jobname = "Systematics_$(conduit)_$(depth)_$(radius)_$(ar)_$(extension)"
            diameter = 2*(radius*ar)
            str =
"#!/bin/bash -l
#SBATCH --job-name=\"$(jobname)\"
#SBATCH --nodes=1
#SBATCH --output=out_vep.o
#SBATCH --error=er_vep.e
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --account c23

srun julia --project=. -O3 --startup-file=no --check-bounds=no SmallScaleCaldera/Caldera2D.jl $(conduit) $(depth) $(radius) $(ar) $(extension)"
            if diameter <= 5.0
                open("runme_test.sh", "w") do io
                    println(io, str)
                end

                # Submit the job
                # run(`sbatch runme_test.sh`)
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
