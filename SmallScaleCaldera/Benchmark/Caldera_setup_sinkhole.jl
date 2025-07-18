## Model Setup
using GeophysicalModelGenerator

function setup2D(
        nx, nz;
        sticky_air = 5.0e0,
        dimensions = (30.0e0, 20.0e0), # extent in x and y in km
        flat = true,
        chimney = false,
        layers = 1,
        volcano_size = (3.0e0, 5.0e0),
        conduit_radius = 0.2,
        chamber_T = 1.0e3,
        chamber_depth = 5.0e0,
        chamber_radius = 2.5e0,
        aspect_x = 1.0,
    )

    Lx = Ly = dimensions[1]
    x = range(0.0, Lx, nx)
    y = range(0.0, Ly, 2)
    z = range(-dimensions[2], sticky_air, nz)
    Grid = CartData(xyz_grid(x, y, z))

    # Allocate Phase and Temp arrays
    air_phase = layers + 6
    # Phases = fill(6, nx, 2, nz);
    Phases = fill(air_phase, nx, 2, nz)
    Temp = fill(0.0, nx, 2, nz)
    Temp_bg = fill(0.0, nx, 2, nz)

    add_box!(
        Phases, Temp, Grid;
        xlim = (minimum(Grid.x.val), maximum(Grid.x.val)),
        ylim = (minimum(Grid.y.val), maximum(Grid.y.val)),
        zlim = (minimum(Grid.z.val), 0.0),
        phase = LithosphericPhases(Layers = [chamber_depth], Phases = [1, 2]),
        T = HalfspaceCoolingTemp(Age = 20)
    )

    add_stripes!(Phases, Grid;
        stripAxes = (0,0,1),
        phase = ConstantPhase(1),
        stripePhase = ConstantPhase(layers + 5),
        # stripeWidth=0.002,
        stripeSpacing=0.5
    )

    if !flat
        heights = layers > 1 ? [volcano_size[1] - i * (volcano_size[1] / layers) for i in 0:(layers - 1)] : volcano_size[1]
        for (i, height) in enumerate(heights)
            add_volcano!(
                Phases, Temp, Grid;
                volcanic_phase = i + 4,  # Change phase for each layer
                center = (mean(Grid.x.val), 0.0),
                height = height,
                radius = volcano_size[2],
                crater = 0.5,
                base = 0.0,
                background = nothing,
                # T               = HalfspaceCoolingTemp(Age=20)
                T = i == 1 ? HalfspaceCoolingTemp(Age = 20) : nothing,
            )
        end
    end

    Temp_bg = copy(Temp)

    add_ellipsoid!(
        Phases, Temp, Grid;
        cen = (mean(Grid.x.val), 0, -chamber_depth),
        axes = (chamber_radius * aspect_x, 2.5, chamber_radius),
        phase = ConstantPhase(3),
        T = ConstantTemp(T = chamber_T - 100.0e0)
    )

    add_ellipsoid!(
        Phases, Temp, Grid;
        cen = (mean(Grid.x.val), 0, -(chamber_depth - (chamber_radius / 2))),
        axes = ((chamber_radius / 1.25) * aspect_x, 2.5, (chamber_radius / 2)),
        phase = ConstantPhase(4),
        T = ConstantTemp(T = chamber_T)
    )


    Grid = addfield(Grid, (; Phases, Temp))
    li = (abs(last(x) - first(x)), abs(last(z) - first(z))) .* 1.0e3
    origin = (x[1], z[1]) .* 1.0e3

    ph = Phases[:, 1, :]
    T = Temp[:, 1, :] .+ 273
    T_bg = Temp_bg[:, 1, :] .+ 273
    V_total = (4 / 3 * π * (chamber_radius * aspect_x) * chamber_radius * (chamber_radius * aspect_x)) * 1.0e9
    V_eruptible = (4 / 3 * π * (chamber_radius / 1.25) * aspect_x * (chamber_radius / 2) * ((chamber_radius / 1.25) * aspect_x)) * 1.0e9
    R = ((chamber_depth - chamber_radius)) / (chamber_radius * aspect_x)
    chamber_diameter = 2 * (chamber_radius * aspect_x)
    chamber_erupt = 2 * ((chamber_radius / 1.25) * aspect_x)
    printstyled("Magma volume of the initial chamber:$(round(ustrip.(uconvert(u"km^3", (V_total)u"m^3")); digits = 5)) km³ \n"; bold = true, color = :red, blink = true)
    printstyled("Eruptible magma volume: $(round(ustrip.(uconvert(u"km^3", (V_eruptible)u"m^3")); digits = 5)) km³ \n"; bold = true, color = :red, blink = true)
    printstyled("Roof ratio (Depth/half-axis width): $R \n"; bold = true, color = :cyan)
    printstyled("Chamber diameter: $(round(chamber_diameter; digits = 3)) km \n"; bold = true, color = :light_yellow)
    printstyled("Eruptible chamber diameter: $(round(chamber_erupt; digits = 3)) km \n"; bold = true, color = :light_yellow)
    # write_paraview(Grid, "Volcano2D")
    return li, origin, ph, T, T_bg, Grid, V_total, V_eruptible, layers, air_phase
end
