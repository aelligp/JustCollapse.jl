function main()

    conduit, depth, radius, ar, extension = 0.2e0, 5e0, 1.25e0, 2, 1e-15

    run(`$(Base.julia_cmd()) --project=. -O3 --startup-file=no --check-bounds=no miniapps/benchmarks/stokes2D/Volcano2D/Caldera2D.jl $(conduit) $(depth) $(radius) $(ar) $(extension)`)

end

main()
