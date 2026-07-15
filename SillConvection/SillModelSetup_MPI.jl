## Model Setup
using GeophysicalModelGenerator

function SillSetup_MPI(nx, nz, xvi;
    dimensions = (0.3, 0.25), # extent in x and y in km
    sill_temp = 1000,       # temperature in C
    host_rock_temp = 500,
    sill_size = 0.1, # size of the sill in km
    )

    Lx = Ly = dimensions[1]
    x = range(minimum(xvi[1]), maximum(xvi[1]), nx)
    println("x: ", x)
    y = range(0.0, Ly, 2)
    z = range(minimum(xvi[2]), maximum(xvi[2]), nz)
    println("z: ", z)
    Grid = CartData(xyz_grid(x, 0, z))
    sill_placement = (dimensions[2] - (dimensions[2] - sill_size)/2, 0.0 + (dimensions[2] - sill_size)/2)
    println("sill_placement: ", sill_placement)
    Phases = fill(1, nx, 1, nz)
    Temp = fill(host_rock_temp, nx, 1, nz)

    add_box!(
        Phases,
        Temp,
        Grid;
        xlim = (minimum(Grid.x.val), maximum(Grid.x.val)),
        ylim = (minimum(Grid.y.val), maximum(Grid.y.val)),
        zlim = (-sill_placement[1], -sill_placement[2]),
        # zlim = (-0.175, -0.075),
        phase=ConstantPhase(2),
        T = ConstantTemp(; T=sill_temp)
    )

    # add_sphere!(Phases, Temp, Grid;
    # cen = (mean(Grid.x.val), 0,-90.0),
    # radius = 5,
    # phase  = ConstantPhase(3),
    # T      = ConstantTemp(T=1100+273)
    # )

    Grid = addfield(Grid, (; Phases, Temp))
    write_paraview(Grid, "SillConvection_$(igg.me)")

    li = (abs(last(x) - first(x)), abs(last(z) - first(z))) .* 1.0e3
    origin = (x[1], z[1]) .* 1.0e3

    ph = Phases[:, 1, :]
    T = Temp[:, 1, :] .+ 273

    return li, origin, ph, T, Grid
end
