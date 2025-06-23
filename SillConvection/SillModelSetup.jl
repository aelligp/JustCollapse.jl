## Model Setup
using GeophysicalModelGenerator

function SillSetup(nx, nz;
    dimensions = (0.3, 0.25), # extent in x and y in km
    sill_temp = 1000,       # temperature in C
    host_rock_temp = 500,
    sill_size = 0.1, # size of the sill in km
    ellipse = false
    )
    Lx = Ly = dimensions[1]
    x = range(0.0, Lx, nx)
    y = range(0.0, Ly, 2)
    z = range(-dimensions[2], 0.0, nz)
    Grid = CartData(xyz_grid(x, y, z))
    sill_placement = (dimensions[2] - (dimensions[2] - sill_size)/2, 0.0 + (dimensions[2] - sill_size)/2)

    Phases = fill(1, nx, 2, nz)
    Temp = fill(host_rock_temp, nx, 2, nz)

    if ellipse
        # Elliptical sill
        add_ellipsoid!(
            Phases, Temp, Grid;
            cen = (mean(Grid.x.val), 0.0, mean(Grid.z.val)),
            axes = (sill_size, dimensions[1] * 0.75, sill_size * 0.5),
            phase = ConstantPhase(2),
            T = ConstantTemp(T = sill_temp)
        )
    else
        add_box!(
            Phases,
            Temp,
            Grid;
            xlim = (minimum(Grid.x.val), maximum(Grid.x.val)),
            ylim = (minimum(Grid.y.val), maximum(Grid.y.val)),
            zlim = (-sill_placement[1], -sill_placement[2]),
            phase=ConstantPhase(2),
            T = ConstantTemp(; T=sill_temp)
        )
    end

    # add_sphere!(Phases, Temp, Grid;
    # cen = (mean(Grid.x.val), 0,-90.0),
    # radius = 5,
    # phase  = ConstantPhase(3),
    # T      = ConstantTemp(T=1100+273)
    # )

    Grid = addfield(Grid, (; Phases, Temp))
    li = (abs(last(x) - first(x)), abs(last(z) - first(z))) .* 1.0e3
    origin = (x[1], z[1]) .* 1.0e3

    ph = Phases[:, 1, :]
    T = Temp[:, 1, :] .+ 273

    return li, origin, ph, T, Grid
end
