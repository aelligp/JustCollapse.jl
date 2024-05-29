using Pkg
Pkg.activate(".")
using JustRelax
# using MagmaThermoKinematics
ENV["PS_PACKAGE"] = :Threads     # if GPU use :CUDA

const USE_GPU = false;
const Topography = true; #specify if you want topography plotted in the figures

if USE_GPU
    model = PS_Setup(:gpu, Float64, 2)            # initialize parallel stencil in 2D
    environment!(model)
else
    model = PS_Setup(:cpu, Float64, 2)            # initialize parallel stencil in 2D
    environment!(model)
end

using Printf, LinearAlgebra, GeoParams, CairoMakie
using ParallelStencil.FiniteDifferences2D   #specify Dimension
using ImplicitGlobalGrid
using MPI: MPI
using GeophysicalModelGenerator, StencilInterpolations, StaticArrays
# using Plots 
using WriteVTK
# using JustPIC

using MagmaThermoKinematics
# -----------------------------------------------------------------------------------------
# Viscosity calculation functions stored in this script
include("./src/Helperfunc.jl");
include("./src/LoadModel.jl");
println("Loaded helper functions")
#------------------------------------------------------------------------------------------
LonLat = load("./Data/ExportToba_2.jld2", "TobaTopography_LATLON")#, "TobaTopography_Cart");

proj = ProjectionPoint(; Lat=2.19, Lon=98.91);
Topo = Convert2CartData(LonLat, proj);

println("Done loading Model... starting Dike Injection 2D routine")
#------------------------------------------------------------------------------------------
@views function DikeInjection_2D()

    # IO ----- -------------------------------------------
    figdir = "./fig2D/new_solver"
    gifname = "new_solver"
    pvd = paraview_collection("new_solver")
    !isdir(figdir) && mkpath(figdir)
    # anim = Animation(figdir, String[]);
    # Preparation of VTK/Paraview output 
    if isdir("./Paraview/new_solver") == false
        mkdir("./Paraview/new_solver")
    end
    loadpath = "./Paraview/new_solver/"

    # Standard MTK Toba routine--------------------------------------------------------------

    # nondimensionalize with CharDim 
    CharDim = GEO_units(; length=40km, viscosity=1e20Pa * s)

    # dimensions should be multiplication of 32 to be scalable @ GPU
    Nx, Ny, Nz = 100, 100, 100
    #2D grid size
    # should be a multitude 32-2 for optimal GPU perfomance 
    nt = 15           # number of timesteps
    InjectionInterval = 10            # number of timesteps between injections
    nx, ny = (128, 128) .- 2 #update 

    Grid = CreateCartGrid(;
        size=(Nx, Ny, Nz),
        x=((Topo.x.val[1, 1, 1])km, (Topo.x.val[end, 1, 1])km),
        y=((Topo.y.val[1, 1, 1])km, (Topo.y.val[1, end, 1])km),
        z=(-40km, 4km),
    )
    X, Y, Z = XYZGrid(Grid.coord1D...)
    DataTest = CartData(X, Y, Z, (Depthdata=Z,))

    Lon, Lat, Depth = (Topo.x.val .* km), (Topo.y.val .* km), ((Topo.z.val ./ 1e3) .* km)
    Topo_Cart = CartData(Lon, Lat, Depth, (Depth=Depth,))

    ind = AboveSurface(DataTest, Topo_Cart)
    Phase = ones(size(X))
    Phase[ind] .= 3

    DataPara = CartData(X, Y, Z, (Phase=Phase,))
    Phase = Int64.(round.(DataPara.fields.Phase))

    # ----------CrossSections for 2D simulation----------------------------------

    # Data_Cross              = CrossSection(DataPara, dims=(nx+1,ny+1), Interpolate=true,Start=(ustrip(-63.5km),ustrip(80.95km)), End=(ustrip(-03.72km), ustrip(20.05km)))
    Data_Cross = CrossSection(
        DataPara;
        dims=(nx + 1, ny + 1),
        Interpolate=true,
        Start=(ustrip(-50.00km), ustrip(60.00km)),
        End=(ustrip(-06.00km), ustrip(20.00km)),
    )
    x_new = FlattenCrossSection(Data_Cross)
    Data_Cross = AddField(Data_Cross, "FlatCrossSection", x_new)
    Phase = dropdims(Data_Cross.fields.Phase; dims=3) #Dropped the 3rd dim of the Phases to be consistent with new grid of the CrossSection
    Phase = Int64.(round.(Phase))

    ## Workaround to display topo on heatmaps for η,ρ,ϕ due to different dimensions (ni, not ni+1)
    # Data_Cross_ni           = CrossSection(DataPara, dims=(nx,ny), Interpolate=true,Start=(ustrip(Grid.min[1]),ustrip(Grid.max[2])), End=(ustrip(Grid.max[1]), ustrip(Grid.min[2])))
    # Data_Cross_ni           = CrossSection(DataPara, dims=(nx,ny), Interpolate=true,Start=(ustrip(-63.5km),ustrip(80.95km)), End=(ustrip(-03.72km), ustrip(20.05km)))
    Data_Cross_ni = CrossSection(
        DataPara;
        dims=(nx, ny),
        Interpolate=true,
        Start=(ustrip(-50.00km), ustrip(60.00km)),
        End=(ustrip(-06.00km), ustrip(20.00km)),
    )
    Phase_ni = dropdims(Data_Cross_ni.fields.Phase; dims=3) #Dropped the 3rd dim of the Phases to be consistent with new grid of the CrossSection
    Phase_ni = Int64.(round.(Phase_ni))

    #Seismo Model 
    # Model3D_Cross           = CrossSection(Model3D_cart,dims=(nx+1,ny+1), Interpolate=true, Start=(ustrip(Grid.min[1]),ustrip(Grid.max[2])), End=(ustrip(Grid.max[1]), ustrip(Grid.min[2])));
    # Model3D_Cross           = CrossSection(Model3D_cart,dims=(nx+1,ny+1), Interpolate=true, Start=(ustrip(-63.5km),ustrip(80.95km)), End=(ustrip(-03.72km), ustrip(20.05km)))
    Model3D_Cross = CrossSection(
        Model3D_cart;
        dims=(nx + 1, ny + 1),
        Interpolate=true,
        Start=(ustrip(-50.00km), ustrip(60.00km)),
        End=(ustrip(-06.00km), ustrip(20.00km)),
    )
    Model3D_new = FlattenCrossSection(Model3D_Cross)
    Model3D_Cross = AddField(Model3D_Cross, "Model3D_Cross", Model3D_new)

    #New 2D Grid
    Grid2D = CreateCartGrid(;
        size=(nx, ny),
        x=(extrema(Data_Cross.fields.FlatCrossSection) .* km),
        z=((minimum(Data_Cross.z.val) ./ 2) .* km, (maximum(Data_Cross.z.val)) .* km),
        CharDim=CharDim,
    ) #create new 2D grid for Injection routine

    Phi_melt_data = dropdims(Model3D_Cross.fields.Phi_melt; dims=3)

    #Dike location initiation                                                              

    ind_melt = findall(Phi_melt_data .> 0.12) # only Meltfraction of 12% used for dike injection
    x1, z1 = dropdims(Data_Cross.fields.FlatCrossSection; dims=3),
    dropdims(Data_Cross.z.val; dims=3) # Define x,z coord for injection
    x1, z1 = x1 .* km, z1 .* km # add dimension to FlatCrossSection to non-dim x,z for consistency
    x1, z1 = nondimensionalize(x1, CharDim), nondimensionalize(z1, CharDim)

    #----------Paraview Files--------------------------------------------------------------------
    ## Save Paraview Data to visualise 
    # Write_Paraview(Topo_new, "Topography")
    # Write_Paraview(DataPara,"Phase")
    # Write_Paraview(Data_Cross,"2DPhase")
    # Write_Paraview(Data_Cross,"Cross_short")
    # Write_Paraview(Model3D_Cross,"Model3D_Cross")

    #-------rheology parameters--------------------------------------------------------------
    η_uppercrust = 1e21
    η_magma = 1e18
    η_air = 1e21
    η_reg = 1e18
    G0 = 6e10                                                             # shear modulus
    Composite_creep = 1                                                   # if multiple creep laws are used define which one sets the initial viscosity
    pl = DruckerPrager_regularised(; C=Inf, ϕ=30.0, η_vp=η_reg, Ψ=15.0) # non-regularized plasticity
    el = SetConstantElasticity(; G=G0, ν=0.5)                            # elastic spring
    disl_creep = DislocationCreep(; A=10^-15.0, n=2.0, E=476e3, V=0.0, r=0.0, R=8.3145) #AdM    #(;E=187kJ/mol,) Kiss et al. 2023
    diff_creep = DiffusionCreep()
    creep_rock = LinearViscous(; η=η_uppercrust * Pa * s)
    creep_magma = LinearViscous(; η=η_magma * Pa * s)
    creep_air = LinearViscous(; η=η_air * Pa * s)
    creep = LinearViscous(; η=1e20 * Pa * s)
    β = inv(get_Kb(el)) #=0.0/Pa=#

    #-------JustRelax parameters-------------------------------------------------------------
    # Domain setup for JustRelax 
    ni = (nx, ny)   #grid spacing for JustRelax calculation 
    lx, lz = (Grid2D.L[1]), (Grid2D.L[2]) # nondim if CharDim=CharDim
    li = lx, lz
    init_MPI = MPI.Initialized() ? false : true
    finalize_MPI = false
    b_width = (4, 4, 0) #boundary width
    origin = Grid2D.min[1], Grid2D.min[2]
    igg = IGG(init_global_grid(nx, ny, 0; init_MPI=init_MPI)...) #init MPI
    di = @. li / (nx_g(), ny_g())
    xci, xvi = lazy_grid(di, li, ni; origin=origin) #non-dim nodes at the center and the vertices of the cell (staggered grid)
    # xci, xvi                = Grid2D.coord1D_cen, Grid2D.coord1D    #non-dim nodes at the center and the vertices of the cell (staggered grid)
    #---------------------------------------------------------------------------------------
    # Set material parameters                                       
    MatParam = (
        SetMaterialParams(;
            Name="UpperCrust",
            Phase=1,
            Density=PT_Density(; ρ0=3000kg / m^3, β=β),
            HeatCapacity=ConstantHeatCapacity(; cp=1050J / kg / K),
            Conductivity=ConstantConductivity(; k=3.0Watt / K / m),
            LatentHeat=ConstantLatentHeat(; Q_L=350e3J / kg),
            #    CreepLaws= LinearViscous(η=η_uppercrust*Pa * s),
            CompositeRheology=CompositeRheology((creep_rock, el)),
            Melting=MeltingParam_Caricchi(),
            Elasticity=el, #ConstantElasticity(; G=Inf*Pa, Kb=Inf*Pa),
            # Elasticity = ConstantElasticity(; G=Inf*Pa, Kb=Inf*Pa),
            CharDim=CharDim,
        ),
        SetMaterialParams(;
            Name="Magma",
            Phase=2,
            Density=PT_Density(; ρ0=2800kg / m^3, β=0.0 / Pa),
            HeatCapacity=ConstantHeatCapacity(; cp=1050J / kg / K),
            Conductivity=ConstantConductivity(; k=3.0Watt / K / m),
            LatentHeat=ConstantLatentHeat(; Q_L=350e3J / kg),
            #   CreepLaws = LinearViscous(η= η_magma*Pa*s),
            CompositeRheology=CompositeRheology((creep_magma,)),
            Melting=MeltingParam_Caricchi(),
            Elasticity=ConstantElasticity(; G=Inf * Pa, Kb=Inf * Pa),
            CharDim=CharDim,
        ),
        SetMaterialParams(;
            Name="Sticky Air",
            Phase=3,
            Density=ConstantDensity(; ρ=2800kg / m^3),
            HeatCapacity=ConstantHeatCapacity(; cp=1050J / kg / K),
            Conductivity=ConstantConductivity(; k=15Watt / K / m),
            LatentHeat=ConstantLatentHeat(; Q_L=0.0J / kg),
            CompositeRheology=CompositeRheology((creep_air,)),
            Elasticity=ConstantElasticity(; G=Inf * Pa, Kb=Inf * Pa),
            # Melting = MeltingParam_Caricchi()
            CharDim=CharDim,
        ),
    )

    # Physical Parameters 

    # ΔT                      =   nondimensionalize(800C, CharDim)
    ΔT = nondimensionalize(600C, CharDim)
    GeoT = -(ΔT - nondimensionalize(0C, CharDim)) / li[2]
    η = MatParam[1].CompositeRheology[1][1].η.val
    Cp = MatParam[2].HeatCapacity[1].cp.val  # heat capacity     
    ρ0 = MatParam[2].Density[1].ρ0.val        # reference Density
    k0 = MatParam[2].Conductivity[1].k.val   # Conductivity
    G = MatParam[1].Elasticity[1].G.val    # Shear Modulus
    κ = k0 / (ρ0 * Cp)                       # thermal diffusivity
    g = MatParam[2].Gravity[1].g.val        # Gravity
    # α                       =   0.03
    α = MatParam[1].Density[1].α.val                     # thermal expansion coefficient for PT Density
    Ra = ρ0 * g * α * ΔT * 10^3 / (η * κ)
    dt = dt_diff = 0.5 / 6.1 * min(di...)^3 / κ # diffusive CFL timestep limiter
    # dt                      =   dt_diff = 0.5 / 2.1 * min(di...)^2 / κ # diffusive CFL timestep limiter
    # -- Dike parameters -----------------------------------------------------
    #  
    W_in, H_in = nondimensionalize(5.0km, CharDim), nondimensionalize(0.5km, CharDim) # Width and thickness of dike
    T_in = nondimensionalize(900C, CharDim)
    H_ran, W_ran = length(z1), length(x1) # Size of domain in which we randomly place dikes and range of angles
    Dike_Type = "ElasticDike"
    Tracers = StructArray{Tracer}(undef, 1)
    nTr_dike = 300
    # center not used in this case as we pass the location of the seismic inversion
    # center                  = @. getindex.(xvi,1) + li* 0.5;      
    # --------------------------------------------------------

    phase_v = Int64.(PTArray(ones(ni .+ 1...)))        # constant for now
    phase_c = Int64.(PTArray(ones(ni...)))           # constant for now
    if Topography == true
        for i in CartesianIndices(phase_v)
            # vx, vz = xvi[1][i[1]], xvi[2][i[2]]
            if Phase[i] == 3
                phase_v[i] = 3
            end
        end
        for i in CartesianIndices(phase_c)
            # cx, cz = xci[1][i[1]], xci[2][i[2]]
            if Phase[i] == 3
                phase_c[i] = 3
            end
        end
    else
        let
            Yv = [y for x in xvi[1], y in xvi[2]]
            Yc = [y for x in xci[1], y in xci[2]]
            mask = Yv .> 0
            @views phase_v[mask] .= 3
            mask = Yc .> 0
            @views phase_c[mask] .= 3
        end
    end

    # r_sphere = ustrip.(nondimensionalize(2.5km, CharDim))
    #  for i in CartesianIndices(phase_v)
    # vx, vy = xvi[1][i[1]], xvi[2][i[2]]
    # li./0.5
    #     # if π*r_sphere^2 < 1.0
    #     #     phase_v[i] = 2
    #     # end

    # end

    #----- thermal Array setup ----------------------------------------------------------------------
    thermal = ThermalArrays(ni)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux=(left=true, right=true, top=false, bot=false),
        periodicity=(left=false, right=false, top=false, bot=false),
    )

    thermal.T .= PTArray([
        xvi[2][iy] * GeoT + nondimensionalize(0C, CharDim) for ix in axes(thermal.T, 1),
        iy in axes(thermal.T, 2)
    ])
    Tnew_cpu = Matrix{Float64}(undef, ni .+ 1...)
    Phi_melt_cpu = Matrix{Float64}(undef, ni...)

    @views thermal.T[:, 1] .= ΔT
    ind = findall(phase_v -> phase_v == 3, phase_v)
    @views thermal.T[ind] .= nondimensionalize(20C, CharDim)
    @views thermal.T[:, end] .= nondimensionalize(20C, CharDim)
    ind = (thermal.T .<= nondimensionalize(0C, CharDim))
    @views thermal.T[ind] .= nondimensionalize(20C, CharDim)
    @views thermal.T[:, end] .= nondimensionalize(20C, CharDim)
    @copy thermal.Told thermal.T
    @copy Tnew_cpu Array(thermal.T[2:(end - 1), :])

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(ni, ViscoElastic)
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-5, CFL=0.27 / √2.1) #ϵ=1e-4,  CFL=1 / √2.1 CFL=0.27 / √2.1
    # Rheology
    η = @ones(ni...)
    args_η = (; T=thermal.T)
    # @hide_communication b_width begin
    # @parallel (@idx ni) initViscosity!(phase_c, MatParam) # init viscosity field
    @parallel (@idx ni) init_Viscosity!(η, phase_c, MatParam) # init viscosity field
    # @parallel (@idx ni) initViscosity_pl!(η, MatParam[1].CompositeRheology[1][1], args_η) # init viscosity field
    # update_halo!(η)
    # end

    η_vep = deepcopy(η)
    G = @fill(MatParam[1].Elasticity[1].G.val, ni...)
    ϕ = similar(η) # melt fraction
    S, mfac = 1.0, -2.8 # factors for hexagons
    #Multiple Phases defined
    η_f = MatParam[2].CompositeRheology[1][1].η.val    # melt viscosity
    # η_f             = 1e18    # melt viscosity
    η_s = MatParam[1].CompositeRheology[1][1].η.val    # solid viscosity
    # η_s             = 1e21    # solid viscosity
    args_η = (; ϕ=ϕ, T=thermal.T)
    # args_ϕ  = (; ϕ = ϕ) #copy paste error???
    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...)
    @parallel (@idx ni) compute_melt_fraction!(ϕ, MatParam, phase_c, (T=thermal.T,))

    compress = 0
    if MatParam[1].Elasticity[1].Kb.val < Inf
        for i in 1:2
            @parallel (@idx ni) compute_ρg!(ρg[2], ϕ, MatParam, (T=thermal.T, P=stokes.P))
            @show compress += 1
        end
    else
        @parallel (@idx ni) compute_ρg!(ρg[2], ϕ, MatParam, (T=thermal.T, P=stokes.P))
    end

    # ----------------------------------------------------
    # Boundary conditions
    flow_bcs = FlowBoundaryConditions(;
        free_slip=(left=true, right=true, top=true, bot=true)
    )
    # ----------------------------------------------------  

    # Time loop
    t, it = 0.0, 0
    InjectVol = 0.0
    local iters

    while it < nt

        # Update buoyancy and viscosity -
        @copy thermal.Told thermal.T
        @parallel (@idx ni) compute_melt_fraction!(ϕ, MatParam, phase_c, (T=thermal.T,))
        @parallel computeViscosity!(η, ϕ, S, mfac, η_f, η_s)
        @copy η_vep η
        @parallel (@idx ni) update_G!(G, MatParam, phase_c)

        # Stokes solver ----------------
        iters = solve!(
            stokes,
            thermal,
            pt_stokes,
            di,
            flow_bcs,
            ϕ,
            ρg,
            η,
            η_vep,
            G,
            phase_v,
            phase_c,
            args_η,
            MatParam, # do a few initial time-steps without plasticity to improve convergence
            dt,   # if no elasticity then Inf otherwise dt
            igg;
            iterMax=10e3,  # 10e3 for testing
            nout=1e3,
            b_width,
            verbose=true,
        )
        dt = compute_dt(stokes, di, dt_diff)
        # ------------------------------
        @show dt

        # Thermal solver ---------------
        solve!(
            thermal,
            thermal_bc,
            stokes,
            phase_v,
            MatParam,
            (; P=stokes.P, T=thermal.T),
            di,
            dt,
        )
        # ------------------------------

        @show it += 1
        t += dt
        if it == 1
            @show Ra
        end
        # # Plotting -----------------------------------------

        # Update buoyancy and viscosity -
        @copy thermal.Told thermal.T

        @parallel (@idx ni) compute_melt_fraction!(ϕ, MatParam, phase_c, (T=thermal.T,))
        @parallel computeViscosity!(η, ϕ, S, mfac, η_f, η_s)

        @copy η_vep η

        Xc, Yc = [x for x in xvi[1], y in xvi[2]], [y for x in xvi[1], y in xvi[2]]
        st = 30                                       # quiver plotting spatial step
        Xp, Yp = Xc[1:st:end, 1:st:end], Yc[1:st:end, 1:st:end]
        Vxp =
            0.5 * (stokes.V.Vx[1:st:(end - 1), 1:st:end] + stokes.V.Vx[2:st:end, 1:st:end])
        Vyp =
            0.5 * (stokes.V.Vy[1:st:end, 1:st:(end - 1)] + stokes.V.Vy[1:st:end, 2:st:end])
        Vscale = 0.5 / maximum(sqrt.(Vxp .^ 2 + Vyp .^ 2)) * di[1] * (st - 1)

        # h1 = heatmap!(ax1, xvi[1], xvi[2], Array(thermal.T) , colormap=:batlow)
        #     h2 = heatmap!(ax2, xci[1], xvi[2], Array(stokes.V.Vy[2:end-1,:]) , colormap=:batlow)
        #     h3 = heatmap!(ax3, xci[1], xci[2], Array(stokes.τ.II) , colormap=:romaO) 
        #     h4 = heatmap!(ax4, xci[1], xci[2], Array(log10.(η)) , colormap=:batlow)

        # Visualization
        # if igg.me == 0 && it == 1 || rem(it, 1) == 0
        if it == 1 || rem(it, 1) == 0
            Vy_c = (stokes.V.Vy[1:(end - 1), :] + stokes.V.Vy[2:end, :]) / 2
            Xp_d = ustrip.(dimensionalize(Xp, km, CharDim))
            Yp_d = ustrip.(dimensionalize(Yp, km, CharDim))

            x_v = ustrip.(dimensionalize(xvi[1], km, CharDim))
            y_v = ustrip.(dimensionalize(xvi[2], km, CharDim))
            x_c = ustrip.(dimensionalize(xci[1], km, CharDim))
            y_c = ustrip.(dimensionalize(xci[2], km, CharDim))

            gather!(Array(thermal.T), Array(thermal.T))
            gather!(Array(stokes.V.Vy), Array(stokes.V.Vy))
            gather!(Array(stokes.τ.II), Array(stokes.τ.II))
            gather!(Array(η), Array(η))
            gather!(Array(ϕ), Array(ϕ))
            gather!(Array(ρg[2]), Array(ρg[2]))
            copy_arrays_GPU2CPU!(Tnew_cpu, Phi_melt_cpu, thermal.T[2:(end - 1), :], ϕ) #T and ϕ  

            T_d = ustrip.(dimensionalize(Array(thermal.T[2:(end - 1), :]), C, CharDim))
            η_d = ustrip.(dimensionalize(Array(η), Pas, CharDim))
            Vy_d = ustrip.(dimensionalize(Array(Vy_c), cm / yr, CharDim))
            # Vy_d= ustrip.(dimensionalize(Array(Velo[2]),   cm/yr, CharDim));
            ρg_d = ustrip.(dimensionalize(Array(ρg[2]), kg / m^3 * m / s^2, CharDim))
            ρ_d = ρg_d / 10
            ϕ_d = Array(ϕ)
            τII_d = ustrip.(dimensionalize(Array(stokes.τ.II), MPa, CharDim))
            t_yrs = dimensionalize(t, yr, CharDim)
            t_Kyrs = t_yrs / 1e3
            t_Myrs = t_Kyrs / 1e3

            ind_topo = findall(phase_v .== 3)

            ind_ni = findall(phase_c .== 3)

            T_d[ind_topo] .= NaN                 #working
            Vy_d[ind_topo] .= NaN                 #working
            η_d[ind_ni] .= NaN
            ϕ_d[ind_ni] .= NaN
            ρ_d[ind_ni] .= NaN
            τII_d[ind_ni] .= NaN

            fig = Figure(;
                resolution=(2000, 2000), createmissing=true, title="t = $(t_Myrs/1e3) Kyrs"
            )
            ar = 2.0
            ar = DataAspect()

            ax0 = Axis(
                fig[1, 1:2];
                aspect=ar,
                title="t = $((ustrip.(t_Kyrs))) Kyrs",
                titlesize=30,
                height=0.0,
            )
            ax0.ylabelvisible = false
            ax0.xlabelvisible = false
            ax0.xgridvisible = false
            ax0.ygridvisible = false
            ax0.xticksvisible = false
            ax0.yticksvisible = false
            ax0.yminorticksvisible = false
            ax0.xminorticksvisible = false
            ax0.xgridcolor = :white
            ax0.ygridcolor = :white
            ax0.ytickcolor = :white
            ax0.xtickcolor = :white
            ax0.yticklabelcolor = :white
            ax0.xticklabelcolor = :white
            ax0.yticklabelsize = 0
            ax0.xticklabelsize = 0
            ax0.xlabelcolor = :white
            ax0.ylabelcolor = :white

            ax1 = Axis(fig[2, 1][1, 1]; aspect=ar, title=L"T [\mathrm{C}]", titlesize=20)
            ax2 = Axis(
                fig[2, 2][1, 1];
                aspect=ar,
                title=L"\log_{10}(\eta [\mathrm{Pas}])",
                titlesize=20,
            )
            ax3 = Axis(
                fig[3, 1][1, 1]; aspect=ar, title=L"Vy [\mathrm{cm/yr}]", titlesize=20
            )
            ax4 = Axis(
                fig[3, 2][1, 1];
                aspect=ar,
                title=L"\rho [\mathrm{kgm}^{-3}]",
                xlabel="Width [km]",
                titlesize=20,
            )
            ax5 = Axis(
                fig[4, 1][1, 1]; aspect=ar, title=L"\phi", xlabel="Width [km]", titlesize=20
            )
            # ax6 = Axis(fig[4,2][1,1], aspect = ar, title = "τII", xlabel="Width [km]")

            linkyaxes!(ax1, ax2)
            hideydecorations!(ax2; grid=false)

            linkyaxes!(ax3, ax4)
            hideydecorations!(ax4; grid=false)
            # linkyaxes!(ax5,ax6)
            # hideydecorations!(ax6, grid = false)

            hidexdecorations!(ax1; grid=false)
            hidexdecorations!(ax3; grid=false)
            hidexdecorations!(ax2; grid=false)
            hidexdecorations!(ax4; grid=false)

            p1 = heatmap!(ax1, x_v, y_v, T_d; colormap=:batlow)
            p2 = heatmap!(ax2, x_c, y_c, log10.(η_d); Colormap=:roma)
            p3 = heatmap!(ax3, x_c, y_c, Vy_d; colormap=:vik)
            p4 = heatmap!(
                ax4,
                x_c,
                y_c,
                ρ_d;
                colormap=:jet,
                xlims=(20.0, 55.0),
                ylims=(-20.0, maximum(y_v)),
            )
            arrows!(
                ax4,
                Xp_d[:],
                Yp_d[:],
                Vxp[:] * Vscale,
                Vyp[:] * Vscale;
                arrowsize=10,
                lengthscale=30,
                arrowcolor=:white,
                linecolor=:white,
            )
            p5 = heatmap!(
                ax5,
                x_c,
                y_c,
                ϕ_d;
                colormap=:lajolla,
                xlims=(20.0, 55.0),
                ylims=(-20.0, maximum(y_v)),
            )
            # p6  = heatmap!(ax6, x_c, y_c, τII_d, colormap=:romaO, label="τII")
            Colorbar(fig[2, 1][1, 2], p1; height=Relative(0.5))
            Colorbar(fig[2, 2][1, 2], p2; height=Relative(0.5))
            Colorbar(fig[3, 1][1, 2], p3; height=Relative(0.5))
            Colorbar(fig[3, 2][1, 2], p4; height=Relative(0.5))
            Colorbar(fig[4, 1][1, 2], p5; height=Relative(0.5))
            # Colorbar(fig[4,2][1,2],p6, label=L"\tau_{\textrm{II}} \textrm{[MPa]}", height=Relative(0.5))
            limits!(ax1, 20.0, 55.0, minimum(y_v), maximum(y_v))
            limits!(ax2, 20.0, 55.0, minimum(y_v), maximum(y_v))
            limits!(ax3, 20.0, 55.0, minimum(y_v), maximum(y_v))
            limits!(ax4, 20.0, 55.0, minimum(y_c), maximum(y_c))
            limits!(ax5, 20.0, 55.0, minimum(y_c), maximum(y_c))

            rowgap!(fig.layout, 1)
            colgap!(fig.layout, 1)

            fig
            save(joinpath(figdir, "$(Int32(it)).png"), fig)
            # end
            # scene = Scene(layout = (3, 2), resolution = (1200, 1200))

            # p1 = heatmap!(
            #     fig[1,1],
            #     x_v,
            #     y_v,
            #     T_d';
            #     aspect_ratio=ar,
            #     xlims = extrema(x_v),
            #     ylims = extrema(y_v),
            #     zlims=(0, 900),
            #     c=:batlow,
            #     title="time=$(round(ustrip.(t_Myrs), digits=5)) Myrs",
            #     titlefontsize = 20,
            #     colorbar_title = "\nT [C]",
            #     colorbar_titlefontsize = 12,
            # )

            # p2 = heatmap!(
            #     scene[1,2],
            #     x_c,
            #     y_c,
            #     log10.(η_d');
            #     aspect_ratio=1,
            #     xlims=(minimum(x_v), maximum(x_v)),
            #     ylims=(minimum(y_v), maximum(y_v)),
            #     c=:roma,
            #     colorbar_title="log10(η [Pas])",
            #     colorbar_titlefontsize = 12,
            #     xlabel="width [km]",
            # )
            # #clim=(-4, 4),

            # p3 = heatmap!(scene[2,1], x_v, y_v, Vy_d', aspect_ratio=1, xlims=(20.0, 55.0), ylims=(-20.0, maximum(y_v)), c=:jet,  
            #                         title="Vy [cm/yr]", xlabel="width [km]")

            # p4 = heatmap!(scene[2,2], x_c, y_c, ρ_d', aspect_ratio=1, xlims=(20.0, 55.0), ylims=(-20.0, maximum(y_c)), c=:jet,
            #                         title="ρ [kg/m³]", xlabel="width [km]")

            # # p4 = Plots.quiver!(Xp_d[:], Yp_d[:], quiver=(Vxp[:]*Vscale, Vyp[:]*Vscale), lw=1, c=:white)
            # p4 = arrows!(scene[2,2], Xp_d[:], Yp_d[:], Vxp[:]*Vscale, Vyp[:]*Vscale, arrowsize = 10, lengthscale=1,arrowcolor=:white ,linecolor=:white)

            # p5 = heatmap!(scene[3,1] ,x_c, y_c, ϕ', aspect_ratio=1, xlims=(20.0, 55.0), ylims=(-20.0, maximum(y_v)), c=:jet,
            #                         title="Melt fraction", xlabel="width [km]")

            # # Plots.plot(p1, p2 , p3, p4, p5, layout=(3, 2), size=(1200,1200))

            # frame(anim)

            # display( quiver!(Xp[:], Yp[:], quiver=(Vxp[:]*Vscale, Vyp[:]*Vscale), lw=0.1, c=:blue) )

            # T_d3D = reshape(T_d, (size(T_d)...,1))
            # η_d3D = reshape(η_d, (size(η_d)...,1))
            # ϕ_3D  = reshape(ϕ, (size(ϕ)...,1))
            # Vy_d3D= reshape(Vy_d, (size(Vy_d)...,1))
            # ρ_d3D = reshape(ρ_d, (size(ρ_d)...,1))

            vtkfile = vtk_grid("$loadpath" * "_$(Int32(it+1e4))", xvi[1], xvi[2]) # 2-D VTK file
            vtkfile["Temperature"] = Array(T_d)
            vtkfile["MeltFraction"] = Array(ϕ)
            vtkfile["Viscosity"] = Array(η_d)
            vtkfile["Vy"] = Array(Vy_d)
            vtkfile["Density"] = Array(ρ_d)
            vtkfile["Density"] = Array(ρ_d)               # Store fields in file
            outfiles = vtk_save(vtkfile)
            pvd[ustrip.(t_Myrs)] = vtkfile #=pvd[time/kyr] = vtkfile=#

            # # x2,y2,z2         =   Grid.coord1D[1], Grid.coord1D[2], Grid.coord1D[3]
            # vtkfile = vtk_grid("$loadpath"*"_$(Int32(it+1e4))", Data_Cross.x.val, Data_Cross.y.val, Data_Cross.z.val) # 3-D VTK file
            # vtkfile["Temperature"] = Array(T_d3D); vtkfile["MeltFraction"] = Array(ϕ_3D); 
            # vtkfile["Viscosity"] = Array(η_d3D); vtkfile["Vy"] = Array(Vy_d3D); vtkfile["Density"] = Array(ρ_d3D);               # Store fields in file
            # outfiles = vtk_save(vtkfile); #=pvd[time/kyr] = vtkfile=# pvd[ustrip.(t_Myrs)] = vtkfile                                  # Save file & update pvd file
        end

        # Inject Dike 
        if mod(it, InjectionInterval) == 0
            ID = rand(ind_melt)
            cen = [x1[ID], z1[ID]]  # Randomly vary center of dike
            #   Dike injection based on Toba seismic inversion
            if cen[end] < ustrip(nondimensionalize(-25km, CharDim))
                Angle_rand = [rand(80.0:0.1:100.0)] # Orientation: near-vertical @ depth             
            else
                Angle_rand = [rand(-10.0:0.1:10.0)] # Orientation: near-vertical @ shallower depth     
            end

            #   # General center for dike injection

            #   if center[end] < ustrip(nondimensionalize(-25km, CharDim))
            #     Angle_rand = [rand(80.0:0.1:100.0)] # Orientation: near-vertical @ depth             
            #   else
            #     Angle_rand = [rand(-10.0:0.1:10.0)] # Orientation: near-vertical @ shallower depth     
            #   end
            dike = Dike(;
                Angle=Angle_rand,
                W=W_in,
                H=H_in,
                Type=Dike_Type,
                T=T_in,
                Center=cen,
                Phase=2,
            )
            @copy Tnew_cpu Array(@views thermal.T[2:(end - 1), :])
            ## MTK injectDike
            Tracers, Tnew_cpu, Vol, dike_poly, Velo = MagmaThermoKinematics.InjectDike(
                Tracers, Tnew_cpu, xvi, dike, nTr_dike
            )   # Add dike, move hostrocks

            # copy_arrays_GPU2CPU!(Tnew_cpu, Phi_melt_cpu, thermal.T[2:end-1,:], ϕ)     # Copy arrays to CPU to update properties
            UpdateTracers_T_ϕ!(Tracers, xvi, xci, Tnew_cpu, Phi_melt_cpu)      # Update info on tracers 

            @parallel assign!((@views thermal.Told[2:(end - 1), :]), PTArray(Tnew_cpu))
            @parallel assign!((@views thermal.T[2:(end - 1), :]), PTArray(Tnew_cpu))

            # @parallel assign!((@views stokes.V.Vy[2:end-1,:]), PTArray(Velo[2]))     

            #   add flux to injection
            InjectVol += Vol
            # t_Myrs = dimensionalize(t, Myrs, CharDim)                                                              # Keep track of injected volume
            # println("Added new dike; total injected magma volume = $(round(InjectVol/km³,digits=2)) km³; rate Q=$(round(InjectVol/(t_Myrs),digits=2)) m³/s")
            println(
                "injected dike; total injected magma volume = $(dimensionalize(InjectVol,km^3,CharDim)); Recharge rate: $(dimensionalize(Vol,km^3,CharDim)/(dimensionalize(dt,yr,CharDim)*InjectionInterval))",
            )
        end
    end
    # Plots.gif(anim, "$gifname"*".gif", fps = 1)
    vtk_save(pvd)

    finalize_global_grid(; finalize_MPI=true)

    # return (ni=ni, xci=xci, li=li, di=di), thermal
    return nothing
end

# figdir = "figs2D"
# ni, xci, li, di, figdir ,thermal = DikeInjection_2D();
DikeInjection_2D();
