## Model Setup
# using GLMakie
using GeophysicalModelGenerator, GMT
using Interpolations

function Tanzania_Topo(nx,ny,nz)
    #import topo
    # Topo = import_topo([35.1, 36.56, -3.447, -2.62], file="@earth_relief_03s")
    Topo = load("./Data/Tanzania.jld2", "Tanzania") # 1s resolution

    # extract 2D array with the topographic elevation
    surf = Topo.depth.val[:,:,1]
    # l = maximum(abs.(Topo.depth.val))
    proj = ProjectionPoint(; Lat=-2.52, Lon=36.56);
    cartesian = convert2CartData(Topo, proj)

   # Define the bounds for longitude and latitude
   lon_min = minimum(Topo.lon.val)
   lon_max = maximum(Topo.lon.val)
   lat_min = minimum(Topo.lat.val)
   lat_max = maximum(Topo.lat.val)

   # Ensure the coordinates are within the bounds
   x = LinRange(max(lon_min, 35.1), min(lon_max, 36.42), nx) |> collect
   y = LinRange(max(lat_min, -3.447), min(lat_max, -2.52), ny) |> collect
    X = (
        Topo.lon.val[:, 1, 1],
        Topo.lat.val[1, :, 1],
    )

    write_paraview(Topo,"Tanzania")

    # interpolate from GMT's surface to the resolution nx × ny resolution of our grid
    itp = interpolate(X, surf, Gridded(Linear()))
    surf_interp = [itp(x, y) for x in x, y in y]

    # compute the x and y size of our cartesian model
    lat_dist  = extrema(X[1]) |> collect |> diff |> first |> abs
    long_dist = extrema(X[2]) |> collect |> diff |> first |> abs
    Lx        = lat_dist * 110.574
    Ly        = long_dist * 111.320*cos(lat_dist)
    return surf_interp, cartesian, Lx, Ly
end

function Tanzania_setup3D(nx,ny,nz; sticky_air=5)
    topo_tanzania, Topo_cartesian, Lx, Ly = Tanzania_Topo(nx,ny,nz)

    Grid3D = create_CartGrid(;
        size=(nx, ny, nz),
        x=((Topo_cartesian.x.val[1, 1, 1])km, (Topo_cartesian.x.val[end, 1, 1])km),
        y=((Topo_cartesian.y.val[1, 1, 1])km, (Topo_cartesian.y.val[1, end, 1])km),
        z=(-20km, sticky_air*km),
    )

    Grid3D_cart = CartData(xyz_grid(Grid3D.coord1D...));



    # Now we create an integer array that will hold the `Phases` information (which usually refers to the material or rock type in the simulation)
    Phases = fill(4, nx, ny, nz);

    # In many (geodynamic) models, one also has to define the temperature, so lets define it as well
    Temp = fill(20.0, nx, ny, nz);

    lith = LithosphericPhases(Layers=[30], Phases=[1])

    add_box!(Phases, Temp, Grid3D_cart;
        xlim=(minimum(Grid3D_cart.x.val), maximum(Grid3D_cart.x.val)),
        # ylim=(minimum(Grid3D_cart.y.val), maximum(Grid3D_cart.y.val)),
        zlim=(minimum(Grid3D_cart.z.val), maximum(Grid3D_cart.z.val)),
        phase = lith,
        T = HalfspaceCoolingTemp(Age=11)
    )

    # add_volcano!(Phases, Temp, Grid3D_cart;
    #     volcanic_phase  = 1,
    #     center     = (0, 0.0, 0.0),
    #     height     = 3,
    #     radius     = 5,
    #     crater     = 0.5,
    #     base       = 0.0,
    #     # background = topo_tanzania,
    #     T = HalfspaceCoolingTemp(Age=11)
    # )

    # add_ellipsoid!(Phases, Temp, Grid3D_cart;
    #     cen    = (0, 0, -5.0),
    #     axes   = (5, 5, 2.5),
    #     phase  = ConstantPhase(2),
    #     T      = ConstantTemp(T=1000),
    # )
    # add_sphere!(Phases, Temp, Grid3D_cart;
    #     cen = (0, 0, -5.0),
    #     radius = 0.5,
    #     phase  = ConstantPhase(3),
    #     T      = ConstantTemp(T=1200)
    # )

    for k in axes(Grid3D_cart.z.val, 3), j in axes(topo_tanzania, 2), i in axes(topo_tanzania,1)
        if Grid3D_cart.z.val[i, j, k] > topo_tanzania[i, j]
            Phases[i, j, k] = 4
        end
    end


    @. Temp[Phases == 4]  = 0
    @. Temp               = max(Temp, 0)
    Grid3D_cart = addfield(Grid3D_cart,(; Phases, Temp))
    write_paraview(Grid3D_cart, "Tanzania_3D")

    li = (Grid3D.L[1].val, Grid3D.L[2].val, Grid3D.L[3].val)
    origin = (Grid3D.min[1].val, Grid3D.min[2].val, Grid3D.min[3].val)

    ph      = Phases
    T       = Temp

    return li, origin, ph, T, Grid3D_cart
end


# function Tanzania_setup2D(nx,ny,nz; sticky_air=5)
#     topo_tanzania, Topo_cartesian, Lx, Ly = Tanzania_Topo(nx,ny,nz)

#     Grid3D = create_CartGrid(;
#         size=(nx, ny, nz),
#         x=((Topo_cartesian.x.val[1, 1, 1])km, (Topo_cartesian.x.val[end, 1, 1])km),
#         y=((Topo_cartesian.y.val[1, 1, 1])km, (Topo_cartesian.y.val[1, end, 1])km),
#         z=(-20km, sticky_air*km),
#     )

#     Grid3D_cart = CartData(xyz_grid(Grid3D.coord1D...));

#     # Now we create an integer array that will hold the `Phases` information (which usually refers to the material or rock type in the simulation)
#     Phases = fill(1, nx, ny, nz);

#     # In many (geodynamic) models, one also has to define the temperature, so lets define it as well
#     Temp = fill(20.0, nx, ny, nz);

#     lith = LithosphericPhases(Layers=[30], Phases=[1])

#     add_box!(Phases, Temp, Grid3D_cart;
#         xlim=(minimum(Grid3D_cart.x.val), maximum(Grid3D_cart.x.val)),
#         # ylim=(minimum(Grid3D_cart.y.val), maximum(Grid3D_cart.y.val)),
#         zlim=(minimum(Grid3D_cart.z.val), maximum(Grid3D_cart.z.val)),
#         phase = lith,
#         T = HalfspaceCoolingTemp(Age=11)
#     )

#     for k in axes(Grid3D_cart.z.val, 3), j in axes(topo_tanzania, 2), i in axes(topo_tanzania,1)
#         if Grid3D_cart.z.val[i, j, k] > topo_tanzania[i, j]
#             Phases[i, j, k] = 4
#         end
#     end

#     @. Temp[Phases == 4]  = 20
#     @. Temp               = max(Temp, 20)
#     Grid3D_cart = addfield(Grid3D_cart,(; Phases, Temp))

#     ## make a 2D cross section
#     Cross_section = cross_section(
#         Grid3D_cart;
#         dims=(nx, nz),
#         Start=(ustrip(-40.00km), ustrip(50.00km)),
#         End=(ustrip(-06.00km), ustrip(20.00km)),
#     )
#     Toba_Cross_section = cross_section(
#         Topo_cartesian;
#         dims=(nx, nz),
#         Start=(ustrip(-40.00km), ustrip(50.00km)),
#         End=(ustrip(-06.00km), ustrip(20.00km)),
#     )

#     Cross_section.fields.Phases[above_surface(Cross_section, Topo_cartesian)] .= 4;
#     Cross_section.fields.Phases[below_surface(Cross_section, Topo_cartesian)] .= 1;
#     Grid2D = create_CartGrid(;
#         size=(nx, nz),
#         x=(extrema(Cross_section.fields.FlatCrossSection).*km),
#         z=(extrema(Cross_section.z.val).*km),
#     )
#     ## add an ellipsoid
#     add_ellipsoid!(Cross_section.fields.Phases, Cross_section.fields.Temp, Cross_section;
#         cen    = (mean(Cross_section.x.val), 0, -5.0),
#         axes   = (5, 2e3, 2.5),
#         phase  = ConstantPhase(2),
#         T      = ConstantTemp(T=1000),
#     )
#     ## add a sphere
#     add_ellipsoid!(Cross_section.fields.Phases, Cross_section.fields.Temp, Cross_section;
#     cen    = (mean(Cross_section.x.val), 0, -5.0),
#     axes   = (0.5, 2e3, 0.5),
#     phase  = ConstantPhase(3),
#     T      = ConstantTemp(T=1200),
#     )
#     # write_paraview(Cross_section, "Toba_cross")
#     # write_paraview(Grid2D, "2D_Toba_cross")
#     li = (Grid2D.L[1].val, Grid2D.L[2].val)
#     origin = (Grid2D.min[1].val, Grid2D.min[2].val)

#     ph      = Cross_section.fields.Phases[:,:,1]
#     T       = Cross_section.fields.Temp[:,:,1]

#     ## add topography 2D cross section
#     topo1D = Toba_Cross_section.z.val
#     return li, origin, ph, T, topo1D
# end
