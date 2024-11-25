## Model Setup

function volcano_setup2D(
    xvi, nx,nz;
    flat           = true,
    chimney        = false,
    volcano_size   = (3e0, 5e0),
    chamber_T      = 1e3,
    chamber_depth  = 5e0,
    chamber_radius = 2e0,
    aspect_x       = 1.5,
    )
    x = range(minimum(xvi[1]), maximum(xvi[1]), nx);
    z = range(minimum(xvi[2]), maximum(xvi[2]), nz);
    Grid = CartData(xyz_grid(x,0,z));


    # Now we create an integer array that will hold the `Phases` information (which usually refers to the material or rock type in the simulation)
    Phases = fill(5, nx, 1, nz);

    # In many (geodynamic) models, one also has to define the temperature, so lets define it as well
    Temp = fill(0.0, nx, 1, nz);

    add_box!(Phases, Temp, Grid;
        xlim=(minimum(Grid.x.val), maximum(Grid.x.val)),
        ylim=(minimum(Grid.y.val), maximum(Grid.y.val)),
        zlim=(minimum(Grid.z.val), 0.0),
        phase = LithosphericPhases(Layers=[chamber_depth], Phases=[1, 2]),
        T = HalfspaceCoolingTemp(Age=20)
    )

    if !flat
        add_volcano!(Phases, Temp, Grid;
            volcanic_phase  = 1,
            center          = (mean(Grid.x.val), 0.0),
            height          = volcano_size[1],
            radius          = volcano_size[2],
            # crater          = 0.5,
            base            = 0.0,
            background      = nothing,
            T               = HalfspaceCoolingTemp(Age=20)
        )
    end

    add_ellipsoid!(Phases, Temp, Grid;
        cen    = (mean(Grid.x.val), 0, -chamber_depth),
        axes   = (chamber_radius * aspect_x, 2.5, chamber_radius),
        phase  = ConstantPhase(3),
        T      = ConstantTemp(T=chamber_T)
    )
    add_sphere!(Phases, Temp, Grid;
        cen    = (mean(Grid.x.val), 0, -chamber_depth),
        radius = 1e0,
        phase  = ConstantPhase(4),
        T      = ConstantTemp(T=chamber_T)
    )

    if chimney
        add_cylinder!(Phases, Temp, Grid;
            base = (mean(Grid.x.val), 0, -(chamber_depth-chamber_radius)),
            cap  = (mean(Grid.x.val), 0, volcano_size[1]),
            radius = 0.3,
            phase  = ConstantPhase(3),
            T      = ConstantTemp(T=chamber_T),
        )
    end

    Grid = addfield(Grid,(; Phases, Temp))
    li = (abs(last(x)-first(x)), abs(last(z)-first(z)))
    origin = (x[1], z[1])

    ph      = Phases[:,1,:]
    T       = Temp[:,1,:] .+ 273
    write_paraview(Grid, "Volcano2D")
    return li, origin, ph, T, Grid
end
