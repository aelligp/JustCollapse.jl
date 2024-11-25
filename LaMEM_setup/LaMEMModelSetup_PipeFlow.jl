## Model Setup
function volcano_setup2D(nx,ny,nz;sticky_air=5.0)
    Lx = Ly = 60.0;
    x = range(0., Lx, nx);
    y = range(0.0, Ly, 2);
    z = range(-40, sticky_air, nz);
    Grid = CartData(xyz_grid(x,y,z));


    # fill Phase Array with sticky air phase
    Phases = fill(5, nx, 2, nz);

    # In many (geodynamic) models, one also has to define the temperature, so lets define it as well
    Temp = fill(0.0, nx, 2, nz);

    # Define crust
    add_box!(Phases, Temp, Grid;
        xlim=(minimum(Grid.x.val), maximum(Grid.x.val)),
        ylim=(minimum(Grid.y.val), maximum(Grid.y.val)),
        zlim=(minimum(Grid.z.val), 0.0),
        phase = LithosphericPhases(Layers=[40], Phases=[1]),
        T = LinearTemp()
    )

    #Define mush
    add_ellipsoid!(Phases, Temp, Grid;
        cen    = (mean(Grid.x.val), 0,-7.5),
        axes   = (10.0, 10.0, 5.0),
        phase  = ConstantPhase(2),
        T      = ConstantTemp(T=800)
    )

    # Define Conduit
    add_cylinder!(Phases, Temp, Grid;
    base = (mean(Grid.x.val), 0.0, -5.0),
    cap  = (mean(Grid.x.val), 0, 0.0),
    radius = 0.10,
    phase  = ConstantPhase(4),
    T      = ConstantTemp(T=1000),
    )
    # Define magma
    add_sphere!(Phases, Temp, Grid;
    cen = (mean(Grid.x.val), 0,-5.0),
    radius = 2.5,
    phase  = ConstantPhase(3),
    T      = ConstantTemp(T=1100)
    )


    Grid = addfield(Grid,(; Phases, Temp))


    li = (abs(last(x)-first(x)), abs(last(z)-first(z)))
    origin = (x[1], z[1])

    ph      = Phases[:,1,:]
    T       = Temp[:,1,:]
    # write_paraview(Grid, "Volcano2D")
    return li, origin, ph, T, Grid
end
