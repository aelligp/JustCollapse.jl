# using Pkg
# Pkg.activate(".")
using JustRelax
# ENV["PS_PACKAGE"] = :Threads     # if GPU use :CUDA

const USE_GPU=false;
const GPU_ID = 1;

model = if USE_GPU  
    CUDA.device!(GPU_ID) # select GPU
    PS_Setup(:gpu, Float64, 2)            # initialize parallel stencil in 2D
   
else        
    PS_Setup(:cpu, Float64, 2)            # initialize parallel stencil in 2D
end
environment!(model)                        

using Printf, Statistics, LinearAlgebra, GeoParams, CairoMakie      
using ParallelStencil.FiniteDifferences2D   #specify Dimension
using ImplicitGlobalGrid
using MPI: MPI
using GeophysicalModelGenerator, StencilInterpolations, StaticArrays
# using Plots 
using WriteVTK
# using JustPIC

using MagmaThermoKinematics
using BenchmarkTools
using TimerOutputs

# -----------------------------------------------------------------------------------------
# Viscosity calculation functions stored in this script
include("./src/Helperfunc_old.jl");
include("./src/LoadModel.jl");
println("Loaded helper functions")
#------------------------------------------------------------------------------------------
LonLat                      = load("./Data/ExportToba_2.jld2", "TobaTopography_LATLON");
        
proj                        = ProjectionPoint(; Lat=2.19, Lon=98.91);
Topo                        = Convert2CartData(LonLat, proj);

println("Done loading Model... starting Dike Injection 2D routine")
#------------------------------------------------------------------------------------------
# @views function DikeInjection_2D(igg; figname=figname,nx=nx, ny=ny); 

    #-----------------------------------------------------
    # USER INPUTS
    #-----------------------------------------------------
    Topography= false; #specify if you want topography plotted in the figures
    Inject_Dike = true; #specify if you want to inject a dike
        dike_width = 5km; #specify the width of the dike
        dike_height = 0.5km; #specify the height of the dike
        dike_temp = 900C; #specify the temperature of the dike
    
    toy = false; #specify if you want to use the toy model or the Toba model
        thermal_perturbation = :circular_anomaly! ; #specify if you want a thermal perturbation in the center of the domain or random (:random)
        sphere = 5km;                               #specify the radius of the circular anomaly
        temp_anomaly = 900C;

    regime = ViscoElastoPlastic;                  #Rheology of the Stokes Solver: ViscoElastic, ViscoElastoPlastic
    # figname = "testing";              #name of the figure
    
    # nx, ny                  = (190,190).-2  # 2D grid size - should be a multitude of 32 for optimal GPU perfomance 
    Nx,Ny,Nz                = 100,100,100   # 3D grid size (does not yet matter)
        arrow_steps         = 4;            # number of arrows in the quiver plot

    nt                      = 90             # number of timesteps
    InjectionInterval       = 500yr        # number of timesteps between injections (if Inject_Dike=true). Increase this at higher resolution. 
    
    η_uppercrust            = 1e21          #viscosity of the upper crust
    η_magma                 = 1e18          #viscosity of the magma
    η_air                   = 1e21          #viscosity of the air
    
    #-----------------------------------------------------

    # IO ------------------------------------------------
    figdir = "./fig2D/$figname/";
    # gifname = "new_solver";
    pvd = paraview_collection("./Paraview/$figname");
    !isdir(figdir) && mkpath(figdir);
    # anim = Animation(figdir, String[]);
    # Preparation of VTK/Paraview output 
    if isdir("./Paraview/$figname/")==false mkdir("./Paraview/$figname/") end; loadpath = "./Paraview/$figname/"; 

    # Standard MTK Toba routine--------------------------------------------------------------

    # nondimensionalize with CharDim 
    CharDim                 =    GEO_units(length=40km, viscosity=1e20Pa*s)
    
    Grid                    = CreateCartGrid(size=(Nx,Ny,Nz),x=((Topo.x.val[1,1,1])km,(Topo.x.val[end,1,1])km), y=((Topo.y.val[1,1,1])km,(Topo.y.val[1,end,1])km),z=(-40km,4km))
    X,Y,Z                   = XYZGrid(Grid.coord1D...);
    DataTest                = CartData(X,Y,Z,(Depthdata=Z,));

    Lon,Lat,Depth           = (Topo.x.val.*km), (Topo.y.val.*km), ((Topo.z.val./1e3).*km);
    Topo_Cart                = CartData(Lon,Lat,Depth,(Depth=Depth,));

    ind                     = AboveSurface(DataTest,Topo_Cart);
    Phase                   = ones(size(X));
    Phase[ind]             .= 3;

    DataPara                = CartData(X,Y,Z,(Phase=Phase,));
    Phase                   = Int64.(round.(DataPara.fields.Phase));             

    # ----------CrossSections for 2D simulation----------------------------------

    # Data_Cross              = CrossSection(DataPara, dims=(nx+1,ny+1), Interpolate=true,Start=(ustrip(-63.5km),ustrip(80.95km)), End=(ustrip(-03.72km), ustrip(20.05km)))
    Data_Cross              = CrossSection(DataPara, dims=(nx+1,ny+1), Interpolate=true,Start=(ustrip(-50.00km),ustrip(60.00km)), End=(ustrip(-06.00km), ustrip(20.00km)));
    x_new                   = FlattenCrossSection(Data_Cross);
    Data_Cross              = AddField(Data_Cross,"FlatCrossSection", x_new);
    Phase                   = dropdims(Data_Cross.fields.Phase,dims=3); #Dropped the 3rd dim of the Phases to be consistent with new grid of the CrossSection
    Phase                   = Int64.(round.(Phase));

    ## Workaround to display topo on heatmaps for η,ρ,ϕ due to different dimensions (ni, not ni+1)
    Data_Cross_ni              = CrossSection(DataPara, dims=(nx,ny), Interpolate=true,Start=(ustrip(-50.00km),ustrip(60.00km)), End=(ustrip(-06.00km), ustrip(20.00km)));
    Phase_ni                = dropdims(Data_Cross_ni.fields.Phase,dims=3); #Dropped the 3rd dim of the Phases to be consistent with new grid of the CrossSection
    Phase_ni                = Int64.(round.(Phase_ni));

    #Seismo Model 
    Model3D_Cross              = CrossSection(Model3D_cart, dims=(nx+1,ny+1), Interpolate=true,Start=(ustrip(-50.00km),ustrip(60.00km)), End=(ustrip(-06.00km), ustrip(20.00km)));
    Model3D_new             = FlattenCrossSection(Model3D_Cross);
    
    Model3D_Cross           = AddField(Model3D_Cross,"Model3D_Cross",Model3D_new);
    
    #New 2D Grid
    if Topography == true
    Grid2D                  = CreateCartGrid(size=(nx,ny), x=(extrema(Data_Cross.fields.FlatCrossSection).*km),z=((minimum(Data_Cross.z.val)./2).*km,(maximum(Data_Cross.z.val)).*km), CharDim=CharDim); #create new 2D grid for Injection routine
    else
    # new grid with no topography as the topo now seems to be the problem with strain rate. 
    Grid2D                  = CreateCartGrid(size=(nx,ny), x=(extrema(Data_Cross.fields.FlatCrossSection).*km),z=((minimum(Data_Cross.z.val)./2).*km,0.0.*km), CharDim=CharDim); #create new 2D grid for Injection routine
    end

    Phi_melt_data           = dropdims(Model3D_Cross.fields.Phi_melt, dims=3);
    
    #Dike location initiation                                                              
    
    ind_melt                =  findall(Phi_melt_data.>0.12); # only Meltfraction of 12% used for dike injection
    x1, z1                  = dropdims(Data_Cross.fields.FlatCrossSection,dims=3), dropdims(Data_Cross.z.val,dims=3); # Define x,z coord for injection
    x1, z1                  = x1.*km, z1.*km; # add dimension to FlatCrossSection to non-dim x,z for consistency
    x1, z1                  = nondimensionalize(x1,CharDim), nondimensionalize(z1,CharDim);

    #-------rheology parameters--------------------------------------------------------------
    # plasticity setup
    do_DP   = false               # do_DP=false: Von Mises, do_DP=true: Drucker-Prager (friction angle)
    η_reg   =0.0Pas           # regularisation "viscosity"
    # τ_y     = 35MPa              # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    τ_y     = 0.0MPa              # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    # τ_y     = Inf              # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ       = 30.0*do_DP
    # ϕ       = 1.0*do_DP
    G0      = 1e10Pa             # elastic shear modulus
    Gi      = G0/(6.0-4.0*do_DP) # elastic shear modulus perturbation
    # Coh     = 0.0        # cohesion
    Coh     = τ_y/cosd(ϕ)        # cohesion
    # Coh     = Inf #τ_y/cosd(ϕ)        # cohesion
    
    Composite_creep = 1                                                   # if multiple creep laws are used define which one sets the initial viscosity
    pl = DruckerPrager_regularised(C=Coh, ϕ=ϕ, η_vp=η_reg, Ψ=0)        # non-regularized plasticity
    el = SetConstantElasticity(; G=G0, ν=0.5)                            # elastic spring
    el_magma = SetConstantElasticity(; G=Gi, ν=0.5)                            # elastic spring

    disl_creep = DislocationCreep(A=10^-15.0 , n=2.0, E=476e3, V=0.0  ,  r=0.0, R=8.3145) #AdM    #(;E=187kJ/mol,) Kiss et al. 2023
    diff_creep = DiffusionCreep()
    creep_rock = LinearViscous(; η=η_uppercrust*Pa * s)
    creep_magma = LinearViscous(; η=η_magma*Pa * s)
    creep_air = LinearViscous(; η=η_air*Pa * s)
    β = #=0.0/Pa=# inv(get_Kb(el))
    Kb = get_Kb(el)

    #-------JustRelax parameters-------------------------------------------------------------
    # Domain setup for JustRelax 
    ni                      = (nx,ny)   #grid spacing for JustRelax calculation 
    lx, lz                  = (Grid2D.L[1]), (Grid2D.L[2]) # nondim if CharDim=CharDim
    li                      = lx, lz
    b_width                 = (4, 4, 0) #boundary width
    origin                  = Grid2D.min[1], Grid2D.min[2]
    # igg = if JustRelax.MPI.Initialized()
    #     igg
    # else
    #     IGG(init_global_grid(nx, ny, 0; init_MPI = true)...)
    # end
    igg = igg
    # di                      = @. li / (nx_g(), ny_g()) # grid step in x- and y-direction
    di                      = @. li / (nx, ny) # grid step in x- and y-direction
    xci, xvi                = lazy_grid(di, li, ni; origin=origin) #non-dim nodes at the center and the vertices of the cell (staggered grid)
    #---------------------------------------------------------------------------------------

    # Set material parameters                                       
    MatParam                =   (
        SetMaterialParams(Name="UpperCrust", Phase=1, 
                   Density  = PT_Density(ρ0=3000kg/m^3, β=β),
               HeatCapacity = ConstantHeatCapacity(cp=1050J/kg/K),
               Conductivity = ConstantConductivity(k=1.5Watt/K/m),       
                 LatentHeat = ConstantLatentHeat(Q_L=350e3J/kg),
          CompositeRheology = CompositeRheology((creep_rock, el, pl, )),
        #   CompositeRheology = CompositeRheology((el, creep_rock, pl, )),
                    Melting = MeltingParam_Caricchi(),
                #  Elasticity = ConstantElasticity(; G=Inf*Pa, Kb=Inf*Pa),
                 Elasticity = el,
                   CharDim  = CharDim,), 
  
        SetMaterialParams(Name="Magma", Phase=2, 
                   Density  = PT_Density(ρ0=3000kg/m^3, β=0.0/Pa),               
               HeatCapacity = ConstantHeatCapacity(cp=1050J/kg/K),
               Conductivity = ConstantConductivity(k=1.5Watt/K/m),       
                 LatentHeat = ConstantLatentHeat(Q_L=350e3J/kg),
          CompositeRheology = CompositeRheology((creep_magma,el_magma,)),
        #   CompositeRheology = CompositeRheology((el_magma, creep_magma, pl,)),
                    Melting = MeltingParam_Caricchi(),
                #  Elasticity = ConstantElasticity(; G=Inf*Pa, Kb=Inf*Pa),
                 Elasticity = el_magma,
                   CharDim  = CharDim,),   
                   
        SetMaterialParams(Name="Air", Phase=3, 
                   Density  = PT_Density(ρ0=3000kg/m^3, β=β),
               HeatCapacity = ConstantHeatCapacity(cp=1050J/kg/K),
               Conductivity = ConstantConductivity(k=1.5Watt/K/m),       
                 LatentHeat = ConstantLatentHeat(Q_L=350e3J/kg),
          CompositeRheology = CompositeRheology((creep_rock,)),
                    # Melting = MeltingParam_Caricchi(),
                #  Elasticity = ConstantElasticity(; G=Inf*Pa, Kb=Inf*Pa),
                #  Elasticity = el,
                   CharDim  = CharDim,), 
                                        
        #  SetMaterialParams(Name="Sticky Air", Phase=3, 
        #           Density    = ConstantDensity(ρ=1000kg/m^3),                     
        #         HeatCapacity = ConstantHeatCapacity(cp=1050J/kg/K),
        #         Conductivity = ConstantConductivity(k=15Watt/K/m),       
        #           LatentHeat = ConstantLatentHeat(Q_L=0.0J/kg),
        #            CompositeRheology = CompositeRheology((creep_air,)),
        #            Elasticity = ConstantElasticity(; G=Inf*Pa, Kb=Inf*Pa),
        #                  # Melting = MeltingParam_Caricchi()
        #              CharDim  = CharDim),
                                    )  
                        

    #----------------------------------------------------------------------------------
    # Physical Parameters 
    ΔT                      =   nondimensionalize(600C, CharDim)
    GeoT                    =   -(ΔT - nondimensionalize(0C, CharDim)) / li[2]
    η                       =   MatParam[2].CompositeRheology[1][1].η.val       # viscosity for the Rayleigh number
    # η                       =   MatParam[2].CompositeRheology[1][2].η.val       # viscosity for the Rayleigh number
    Cp0                      =   MatParam[2].HeatCapacity[1].cp.val              # heat capacity     
    ρ0                      =   MatParam[2].Density[1].ρ0.val                   # reference Density
    k0                      =   MatParam[2].Conductivity[1].k.val               # Conductivity
    G                       =   MatParam[1].Elasticity[1].G.val                 # Shear Modulus
    κ                       =   k0/(ρ0 * Cp0);                                   # thermal diffusivity
    g                       =   MatParam[1].Gravity[1].g.val                    # Gravity
    # α                       =   0.03
    α                       =   MatParam[1].Density[1].α.val                    # thermal expansion coefficient for PT Density
    Ra                      =   ρ0 * g * α * ΔT * 10^3 / (η * κ)                # Rayleigh number
    dt                      = dt_diff = 0.5 * min(di...)^2 / κ / 2.01           # diffusive CFL timestep limiter


    # Initialize arrays for PT thermal solver

    ρ = @fill(ρ0, ni...)
    Cp = @fill(Cp0, ni...)
    k = @fill(k0, ni...)
    ρCp = @. Cp * ρ

    pt_thermal = PTThermalCoeffs(k, ρCp, dt, di, li)

    # Initialisation 
    thermal                 = ThermalArrays(ni);                                # initialise thermal arrays and boundary conditions
    thermal_bc              = TemperatureBoundaryConditions(; 
                no_flux     = (left = true , right = true , top = false, bot = false), 
                periodicity = (left = false, right = false, top = false, bot = false),
    );
    Tnew_cpu                = Matrix{Float64}(undef, ni.+1...)                  # Temperature for the CPU
    Phi_melt_cpu            = Matrix{Float64}(undef, ni...);                    # Melt fraction for the CPU
    stokes                  = StokesArrays(ni, regime);                         # initialise stokes arrays with the defined regime
    pt_stokes               = PTStokesCoeffs(li, di; ϵ=1e-5,  CFL=0.8 / √2.1); #ϵ=1e-4,  CFL=1 / √2.1 CFL=0.27 / √2.1

    # Boundary conditions of the flow
    flow_bcs                = FlowBoundaryConditions(; 
                free_slip   = (left=true, right=true, top=true, bot=true), 
    );
    
    # phase_v                 = Int64.(PTArray(ones(ni.+1...)));                  # constant for now
    phase_v                 = Int64.(PTArray(ones(nx.+3,ny.+1)));                  # constant for now
    phase_c                 = Int64.(PTArray(ones(ni...)));                     # constant for now
    η                       = @ones(ni...);                                     # initialise viscosity       
    λ                       = @zeros(ni...);
    η_vep                   = deepcopy(η)                                       # initialise viscosity for the VEP
    G                       = @fill(MatParam[1].Elasticity[1].G.val, ni...)     # initialise shear modulus
    ϕ                       = similar(η)                                        # melt fraction center
    ϕ_v                     = @zeros(ni.+1...)                                  # melt fraction vertices
    S, mfac                 = 1.0, -2.8                                         # factors for hexagons (remnants of the old code, but still used in functions) Deubelbeiss, Kaus, Connolly (2010) EPSL   
    # η_f                     = MatParam[2].CompositeRheology[1][2].η.val         # melt viscosity
    # η_s                     = MatParam[1].CompositeRheology[1][2].η.val         # solid viscosity
    η_f                     = MatParam[2].CompositeRheology[1][1].η.val         # melt viscosity
    η_s                     = MatParam[1].CompositeRheology[1][1].η.val         # solid viscosity

    # Buoyancy force
    ρg                      = @zeros(ni...), @zeros(ni...)                      # ρg[1] is the buoyancy force in the x direction, ρg[2] is the buoyancy force in the y direction                                   
    
    depth                   = PTArray([y for x in xci[1], y in xci[2]])         # depth, not sure what its used for
    
    args_η                  = (; T=thermal.T);                                  # not really necessary anymore, but we keep it for now
    args  = (; ϕ = ϕ,  T = thermal.Tc, P = stokes.P, depth = depth, dt = Inf, mfac=mfac, η_f=η_f, η_s=η_s) # this is used for various functions, remember to update it within loops.

    # Dike parameters 
    W_in, H_in              = nondimensionalize(dike_width, CharDim),nondimensionalize(dike_height, CharDim); # Width and thickness of dike
    T_in                    = nondimensionalize(dike_temp, CharDim);  
    H_ran, W_ran            = length(z1) ,length(x1);                          # Size of domain in which we randomly place dikes and range of angles
    Dike_Type               = "ElasticDike"
    Tracers                 = StructArray{Tracer}(undef, 1);
    nTr_dike                = 300; 
   
    # -Topography -------------------------------------------------------
    if Topography == true
        for i in CartesianIndices(phase_v[2:end-1,:])
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
    # else
    #     let 
    #         Yv = [y for x in xvi[1], y in xvi[2]]
    #         Yc = [y for x in xci[1], y in xci[2]]
    #         mask = Yv .> 0
    #         @views phase_v[mask] .= 3
    #         mask = Yc .> 0
    #         @views phase_c[mask] .= 3
    #     end
    end
    

    #----- thermal Array setup ----------------------------------------------------------------------    
    # thermal.T .= PTArray([xvi[2][iy] * GeoT +
    # nondimensionalize(60C, CharDim) for ix in axes(thermal.T,1), iy in axes(thermal.T,2)]);
    #for the case of no topography:
    thermal.T .= PTArray([xvi[2][iy] * GeoT +
    nondimensionalize(0C, CharDim) for ix in axes(thermal.T,1), iy in axes(thermal.T,2)]);

    @views thermal.T[:, 1] .= ΔT;
    ind                     = findall(phase_v -> phase_v == 3, phase_v);
    @views thermal.T[ind]  .= nondimensionalize((0 .+1e-15)C, CharDim);
    ind                     = (thermal.T.<=nondimensionalize(0C,CharDim));
    @views thermal.T[ind]  .= nondimensionalize((0 .+1e-15)C, CharDim);
    @views thermal.T[:, end] .= nondimensionalize((0 .+1e-15)C, CharDim);
    @copy  thermal.Told thermal.T
    @copy  Tnew_cpu Array(thermal.T[2:end-1,:])

    if toy == true
        if thermal_perturbation == :random
            δT          = 5.0              # thermal perturbation (in %)
            random_perturbation!(thermal.T, δT, (lx*1/8, lx*7/8), (-2000e3, -2600e3), xvi)
            @show "random perturbation"
        elseif thermal_perturbation == :circular
            δT          = 10.0              # thermal perturbation (in %)
            xc, yc      = 0.66*lx, 0.5*lz  # origin of thermal anomaly
            r           = nondimensionalize(5km, CharDim)         # radius of perturbation
            circular_perturbation!(thermal.T, δT, xc, yc, r, xvi)
            @show "circular perturbation"
        else thermal_perturbation == :circular_anomaly
            anomaly     = nondimensionalize(temp_anomaly,CharDim) # temperature anomaly
            xc, yc      = 0.66*lx, 0.4*lz  # origin of thermal anomaly
            radius           = nondimensionalize(sphere, CharDim)         # radius of perturbation
            circular_anomaly!(thermal.T, anomaly, phase_v, xc, yc, radius, xvi)
            circular_anomaly_center!(phase_c, xc, yc, radius, xci)

            @show "circular anomaly"
        end
    end


    @parallel (@idx ni) temperature2center!(thermal.Tc, thermal.T)  
    reset_timer!()
    # @timeit "init_Viscosity!" @parallel (@idx ni) init_Viscosity!(η, phase_c, MatParam) # init viscosity field

    # @parallel (@idx ni) compute_ρg!(ρg[2], args.ϕ, MatParam,(T=thermal.T, P=stokes.P))
   
    @timeit "compute_melt_fraction outside" @parallel (@idx ni) compute_melt_fraction!(args.ϕ, MatParam, phase_c,  (T=thermal.Tc,))

    args            = (; ϕ = ϕ,  T = thermal.Tc, P = stokes.P, depth = depth, dt = Inf) # update args

    # @parallel (@idx ni) compute_ρg!(ρg[2],MatParam, phase_c,(T=thermal.Tc, P=stokes.P))
    #(compute_density!(ρg[2], MatParam, Phases, (P=args.P, T=args.T)) * compute_gravity(MatParam[1]))
        compress = 0; 
        if -Inf < MatParam[1].Elasticity[1].Kb.val < Inf  
            for i in 1:2
            @parallel (@idx ni) compute_ρg!(ρg[2], args.ϕ, MatParam,(T=thermal.Tc, P=stokes.P))
            @parallel init_P!(stokes.P, ρg[2], xci[2])                # init pressure field
   
            @show compress +=1
            end
        else
            @parallel (@idx ni) compute_ρg!(ρg[2], args.ϕ, MatParam,(T=thermal.Tc, P=stokes.P))
            @parallel init_P!(stokes.P, ρg[2], xci[2])                # init pressure field
   
        end
        @parallel (@idx ni) computeViscosity!(η, ϕ, S, mfac, η_f, η_s)
    # ----------------------------------------------------  
   
    # Time loop
    t, it    = 0.0, 0
    interval = 1.0
    InjectVol = 0.0;
    evo_t = Float64[]
    evo_InjVol = Float64[]
    ν = 0.05
    local iters             
    @parallel (@idx ni) update_G!(G, MatParam, phase_c)      
    
    while it < nt   
        # Update buoyancy and viscosity -
        # @copy thermal.Told thermal.T
        # @timeit "compute_melt_while" @parallel (@idx ni) compute_melt_fraction!(ϕ, MatParam, phase_c,  (T=thermal.Tc,))
        # # @timeit "compute_visc_while"  @parallel (@idx ni) compute_viscosity_gp!(η, phase_c, phase_v, args, MatParam)
        # # if it == 1
        # @parallel (@idx ni) compute_ρg!(ρg[2], ϕ, MatParam,(T=thermal.Tc, P=stokes.P))
        # # end
        # # @parallel (@idx ni) compute_viscosity!(η, ν, @strain(stokes)..., args, MatParam)
        # @parallel (@idx ni) computeViscosity!(η, ϕ, S, mfac, η_f, η_s)
        # @parallel (@idx ni) compute_viscosity_MTK!(
        #     η, ν, @strain(stokes)..., args, tupleize(MatParam), phase_c
        # )
        # @copy η_vep η
        # @parallel (@idx ni) update_G!(G, MatParam, phase_c)

        # Stokes solver ----------------
        args  = (; ϕ = ϕ,  T = thermal.Tc, P = stokes.P, depth = depth, dt = Inf, S=S, mfac=mfac, η_f=η_f, η_s=η_s)
        @timeit "Stokes solver" iters = MTK_solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            η, 
            η_vep,
            phase_c,
            args,
            MatParam, # do a few initial time-steps without plasticity to improve convergence
            dt,   # if no elasticity then Inf otherwise dt
            igg;
            iterMax=250e3,  # 10e3 for testing
            nout=1e3,
            b_width,
            verbose=true,
        )
          dt = compute_dt(stokes, di, dt_diff)
        # ------------------------------
       @show dt

       # Inject Dike 
        if Inject_Dike == true
        # if rem(it, 2) == 0
        # if mod(it, InjectionInterval) == 0
        # if mod(ustrip(dimensionalize(t,yr,CharDim)), ustrip(InjectionInterval)) >= 0
            if  ustrip(dimensionalize(t,yr,CharDim)) >= (ustrip(InjectionInterval)*interval) #Inject every time the physical time exceeds the injection interval
                ID        =     rand(ind_melt)
                cen    =     [x1[ID],z1[ID]]  # Randomly vary center of dike
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
                dike      =   Dike(Angle=Angle_rand, W=W_in, H=H_in, Type=Dike_Type, T=T_in, Center=cen, Phase = 2); 
                @copy Tnew_cpu Array(@views thermal.T[2:end-1,:])
                ## MTK injectDike
                Tracers, Tnew_cpu, Vol, dike_poly, Velo  =   MagmaThermoKinematics.InjectDike(Tracers, Tnew_cpu, xvi, dike, nTr_dike);   # Add dike, move hostrocks
        
                @views thermal.Told[2:end-1,:] .= PTArray(Tnew_cpu)
                @views thermal.T[2:end-1,:] .= PTArray(Tnew_cpu)

                # @parallel (@idx size(thermal.Tc)...) temperature2center!(thermal.Tc, thermal.T)
                # @parallel (@idx ni) compute_melt_fraction!(ϕ, MatParam, phase_c,  (T=thermal.Tc,))
                
                copy_arrays_GPU2CPU!(Tnew_cpu, Phi_melt_cpu, thermal.T[2:end-1,:], ϕ)     # Copy arrays to CPU to update properties
                UpdateTracers_T_ϕ!(Tracers, xvi, xci, Tnew_cpu, Phi_melt_cpu);      # Update info on tracers 
        
                
                # @parallel assign!((@views stokes.V.Vy[2:end-1,:]), PTArray(Velo[2]))     

                    #   add flux to injection
                InjectVol +=    Vol
                    # t_Myrs = dimensionalize(t, Myrs, CharDim)                                                              # Keep track of injected volume
                    # println("Added new dike; total injected magma volume = $(round(InjectVol/km³,digits=2)) km³; rate Q=$(round(InjectVol/(t_Myrs),digits=2)) m³/s")
                println("injected dike; total injected magma volume = $(dimensionalize(InjectVol,km^3,CharDim)); Recharge rate: $(dimensionalize(Vol,km^3,CharDim)/(dimensionalize(dt,yr,CharDim)*InjectionInterval))")
                
            end
        end
        if ustrip(dimensionalize(t,yr,CharDim)) > (ustrip(InjectionInterval)*interval)
            interval += 1.0 
        end

    #     # Thermal solver ---------------
    #    @timeit "thermal solver" solve!(
    #         thermal,
    #         thermal_bc,
    #         stokes,
    #         phase_v,
    #         MatParam,
    #         args,
    #         di,
    #         dt,
    #     )
    #     # ------------------------------
    #     # Thermal solver ---------------
       @timeit "thermal solver" heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            MatParam,
            args,
            dt,
            di;
            phase = phase_v,
        ) #crashing at dike injection - phase_v the reason?
    #     # ------------------------------

        # Update buoyancy and viscosity -
        @copy thermal.Told thermal.T
        @parallel (@idx size(thermal.Tc)...) temperature2center!(thermal.Tc, thermal.T)

        # Update buoyancy and viscosity -
        @copy thermal.Told thermal.T
        @timeit "compute_melt_while" @parallel (@idx ni) compute_melt_fraction!(ϕ, MatParam, phase_c,  (T=thermal.Tc,))
        @parallel center2vertex!(ϕ_v,ϕ)
        @parallel update_phase(phase_c, ϕ)
        @parallel update_phase(phase_v[2:end-1,:], ϕ_v)
        # @parallel (@idx ni) compute_ρg!(ρg[2], args.ϕ, MatParam, (T=args.T, P=args.P))
        @parallel (@idx ni) compute_ρg_phase!(ρg[2], phase_c, MatParam, (T=args.T, P=args.P))
        @parallel (@idx ni) computeViscosity!(η, ϕ, S, mfac, η_f, η_s)
        # @copy η_vep η

        @show it += 1
        t += dt
        if it == 1 @show Ra end

        #  # # Plotting -------------------------------------------------------
        # if igg.me == 0 && (it == 1 || rem(it, 1)) == 0
            Xc, Yc    = [x for x=xvi[1], y=xvi[2]], [y for x=xvi[1], y=xvi[2]]
            st        = arrow_steps                                       # quiver plotting spatial step
            Xp, Yp    = Xc[1:st:end,1:st:end], Yc[1:st:end,1:st:end]
            Vxp       = 0.5*(stokes.V.Vx[1:st:end-1,1:st:end  ] + stokes.V.Vx[2:st:end,1:st:end])
            Vyp       = 0.5*(stokes.V.Vy[1:st:end  ,1:st:end-1] + stokes.V.Vy[1:st:end,2:st:end])
            # Vscale    = 0.5/maximum(sqrt.(Vxp.^2 + Vyp.^2)) * di[1]*(st-1)x
        # if it == 1 || rem(it, 1) == 0
            Vy_c = (stokes.V.Vy[1:end-1,:]+stokes.V.Vy[2:end,:])/2
            Vx_c = (stokes.V.Vx[:,1:end-1]+stokes.V.Vx[:,2:end])/2
            Xp_d = ustrip.(dimensionalize(Xp, km, CharDim));
            Yp_d = ustrip.(dimensionalize(Yp, km, CharDim));

            x_v = ustrip.(dimensionalize(xvi[1], km, CharDim))
            y_v = ustrip.(dimensionalize(xvi[2], km, CharDim))
            x_c = ustrip.(dimensionalize(xci[1], km, CharDim))
            y_c = ustrip.(dimensionalize(xci[2], km, CharDim))

            gather!(Array(thermal.T), Array(thermal.T))
            gather!(Array(stokes.V.Vy), Array(stokes.V.Vy))
            gather!(Array(stokes.V.Vx), Array(stokes.V.Vx))
            gather!(Array(stokes.τ.II), Array(stokes.τ.II))
            gather!(Array(stokes.τ.xy), Array(stokes.τ.xy))
            gather!(Array(stokes.ε.II), Array(stokes.ε.II))
            gather!(Array(stokes.ε.xy), Array(stokes.ε.xy))
            gather!(Array(η), Array(η))
            gather!(Array(η_vep), Array(η_vep))
            gather!(Array(ϕ), Array(ϕ))
            gather!(Array(ρg[2]), Array(ρg[2]))
            #what is this for again?
            copy_arrays_GPU2CPU!(Tnew_cpu, Phi_melt_cpu, thermal.T[2:end-1,:], ϕ) #T and ϕ  

            # Average velocity to cell vertices (ghost nodes)
            Vy_v = (stokes.V.Vy[1:end-1,:]+stokes.V.Vy[2:end,:])/2
            Vx_v = (stokes.V.Vx[:,1:end-1]+stokes.V.Vx[:,2:end])/2
            Xp_d = ustrip.(dimensionalize(Xp, km, CharDim));
            Yp_d = ustrip.(dimensionalize(Yp, km, CharDim));

            T_d = ustrip.(dimensionalize(Array(thermal.T[2:end-1,:]), C, CharDim))
            # T_d = ustrip.(dimensionalize(Array(thermal.T), C, CharDim))
            η_d = ustrip.(dimensionalize(Array(η), Pas, CharDim))
            η_vep_d = ustrip.(dimensionalize(Array(η_vep), Pas, CharDim))
            Vy_d= ustrip.(dimensionalize(Array(Vy_v),   cm/yr, CharDim));
            Vx_d= ustrip.(dimensionalize(Array(Vx_v),   cm/yr, CharDim));
            ∇V_d= ustrip.(dimensionalize(Array(stokes.∇V),   cm/yr, CharDim));
            P_d = ustrip.(dimensionalize(Array(stokes.P),   MPa, CharDim));
            # Vy_d= ustrip.(dimensionalize(Array(Velo[2]),   cm/yr, CharDim));
            ρg_d= ustrip.(dimensionalize(Array(ρg[2]),   kg/m^3*m/s^2, CharDim))
            ρ_d = ρg_d/10;
            ϕ_d = Array(ϕ)
            τII_d = ustrip.(dimensionalize(Array(stokes.τ.II),   MPa, CharDim))
            τxy_d = ustrip.(dimensionalize(Array(stokes.τ.xy),   MPa, CharDim))
            τxx_d = ustrip.(dimensionalize(Array(stokes.τ.xx),   MPa, CharDim))
            εII_d = ustrip.(dimensionalize(Array(stokes.ε.II),   s^-1, CharDim))
            εxy_d = ustrip.(dimensionalize(Array(stokes.ε.xy),   s^-1, CharDim))
            t_yrs = dimensionalize(t, yr, CharDim)
            t_Kyrs = t_yrs/1e3
            t_Myrs = t_Kyrs/1e3
            
            
            ind_topo = findall(phase_v.==3)
            ind_ni  = findall(phase_c.==3)

            T_d[ind_topo].= NaN                 
            Vy_d[ind_topo].=NaN                 
            Vx_d[ind_topo].=NaN                 
            η_d[ind_ni].= NaN                   
            η_vep_d[ind_ni].= NaN                   
            ϕ_d[ind_ni]  .= NaN                   
            ρ_d[ind_ni].= NaN
            τII_d[ind_ni].= NaN
            P_d[ind_ni].= NaN

            fig = Figure(resolution = (2000, 2000), createmissing = true)
            ar = 2.0
            ar = DataAspect()

            ax0 = Axis(fig[1,1:2], aspect = ar,title="t = $((ustrip.(t_Kyrs))) Kyrs", titlesize=50, height=0.0)
            ax0.ylabelvisible = false; ax0.xlabelvisible = false; ax0.xgridvisible = false; ax0.ygridvisible = false;
            ax0.xticksvisible = false; ax0.yticksvisible = false; ax0.yminorticksvisible = false; ax0.xminorticksvisible = false;
            ax0.xgridcolor=:white; ax0.ygridcolor=:white; ax0.ytickcolor=:white; ax0.xtickcolor=:white; ax0.yticklabelcolor=:white;
            ax0.xticklabelcolor=:white; ax0.yticklabelsize=0; ax0.xticklabelsize=0; ax0.xlabelcolor=:white; ax0.ylabelcolor=:white;


            ax1 = Axis(fig[2,1][1,1], aspect = ar, title = L"T [\mathrm{C}]", titlesize=40, yticklabelsize=25, xticklabelsize=25, xlabelsize=25)
            # ax2 = Axis(fig[2,2][1,1], aspect = ar, title = L"\log_{10}(\eta [\mathrm{Pas}])",titlesize=40, yticklabelsize=25, xticklabelsize=25, xlabelsize=25)
            ax2 = Axis(fig[2,2][1,1], aspect = ar, title = L"\log_{10}(\eta_{vep} [\mathrm{Pas}])",titlesize=40, yticklabelsize=25, xticklabelsize=25, xlabelsize=25)
            ax3 = Axis(fig[3,1][1,1], aspect = ar, title = L"Vy [\mathrm{cm/yr}]", titlesize=40, yticklabelsize=25, xticklabelsize=25, xlabelsize=25)
            ax4 = Axis(fig[3,2][1,1], aspect = ar, title = L"\rho [\mathrm{kgm}^{-3}]", xlabel="Width [km]", titlesize=40, yticklabelsize=25, xticklabelsize=25, xlabelsize=25)
            # ax4 = Axis(fig[3,2][1,1], aspect = ar, title = L"\varepsilon_{\textrm{xy}}[\mathrm{s}^{-1}]", xlabel="Width [km]", titlesize=40, yticklabelsize=25, xticklabelsize=25, xlabelsize=25)
            ax5 = Axis(fig[4,1][1,1], aspect = ar, title = L"\phi", xlabel="Width [km]", titlesize=40, yticklabelsize=25, xticklabelsize=25, xlabelsize=25)
            ax6 = Axis(fig[4,2][1,1], aspect = ar, title = L"\tau_{\textrm{II}} [MPa]", xlabel="Width [km]", titlesize=40, yticklabelsize=25, xticklabelsize=25, xlabelsize=25)
            # ax6 = Axis(fig[4,2][1,1], aspect = ar, title = L"\tau_{\textrm{xx}} [MPa]", xlabel="Width [km]", titlesize=40, yticklabelsize=25, xticklabelsize=25, xlabelsize=25)
            
            linkyaxes!(ax1,ax2)
            hideydecorations!(ax2, grid = false)

            linkyaxes!(ax3,ax4)
            hideydecorations!(ax4, grid = false)
            # linkyaxes!(ax5,ax6)
            # hideydecorations!(ax6, grid = false)
            
           hidexdecorations!(ax1, grid = false); hidexdecorations!(ax3, grid = false); hidexdecorations!(ax2, grid = false); hidexdecorations!(ax4, grid = false)
            

            p1  = heatmap!(ax1, x_v, y_v, T_d, colormap=:batlow)
            # p2  = heatmap!(ax2, x_c, y_c, log10.(η_d), Colormap=:roma, ) 
            p2  = heatmap!(ax2, x_c, y_c, log10.(η_vep_d), Colormap=:roma, ) 
            # p1 = heatmap!(ax1, x_v, y_v, phase_v, colormap=:jet)
            # p1 = heatmap!(ax1, x_c, y_c, phase_c, colormap=:jet)     
            # p2  = heatmap!(ax2, x_c, y_c, log10.(η_vep_d), Colormap=:roma, )      
            p3  = heatmap!(ax3, x_v, y_v, Vy_d, colormap=:vik) 
            # p3  = heatmap!(ax3, x_v, y_v, sqrt.(Vy_v.^2 .+ Vx_v.^2), colormap=:vik)
            # p3  = heatmap!(ax3, x_c, y_c, ∇V_d, colormap=:vik)
            # p4  = heatmap!(ax4, x_c, y_c, Vx_d, colormap=:vik)
            p4  = heatmap!(ax4, x_c, y_c, ρ_d, colormap=:jet, xlims=(20.0,55.0), ylims=(-20.0, maximum(y_v)))
            # p4  = heatmap!(ax4, x_v, y_v, εxy_d, colormap=:jet)#, xlims=(20.0,55.0), ylims=(-20.0, maximum(y_v)))
                #   arrows!(ax4, Xp_d[:], Yp_d[:], Vxp[:]*Vscale, Vyp[:]*Vscale, arrowsize = 10, lengthscale=30,arrowcolor=:white ,linecolor=:white)
            p5  = heatmap!(ax5, x_c, y_c, ϕ_d, colormap=:lajolla)
            # p5  = heatmap!(ax5, x_c, y_c, P_d, colormap=:lajolla,xlims=(20.0,55.0), ylims=(-20.0, maximum(y_v)))
            p6  = heatmap!(ax6, x_v, y_v, τII_d, colormap=:romaO)
            # p6  = heatmap!(ax6, x_c, y_c, τxy_d, colormap=:romaO, label="\tau_{\textrm{xy}}")
            # p6  = heatmap!(ax6, x_c, y_c, τxx_d, colormap=:romaO, label="\tau_{\textrm{xy}}")
            Colorbar(fig[2,1][1,2], p1, height=Relative(0.7),ticklabelsize=25, ticksize=15)
            Colorbar(fig[2,2][1,2], p2, height=Relative(0.7),ticklabelsize=25, ticksize=15)
            Colorbar(fig[3,1][1,2], p3, height=Relative(0.7),ticklabelsize=25, ticksize=15)
            Colorbar(fig[3,2][1,2], p4,height=Relative(0.7),ticklabelsize=25, ticksize=15)
            Colorbar(fig[4,1][1,2],p5, height=Relative(0.7),ticklabelsize=25, ticksize=15)
            Colorbar(fig[4,2][1,2],p6, height=Relative(0.7),ticklabelsize=25, ticksize=15)
            limits!(ax1, 20.0, 55.0, minimum(y_v), maximum(y_v))
            limits!(ax2, 20.0, 55.0, minimum(y_v), maximum(y_v))
            limits!(ax3, 20.0, 55.0, minimum(y_v), maximum(y_v))
            limits!(ax4, 20.0, 55.0, minimum(y_c), maximum(y_c))
            limits!(ax5, 20.0, 55.0, minimum(y_c), maximum(y_c))
            limits!(ax6, 20.0, 55.0, minimum(y_c), maximum(y_c))
            
            rowgap!(fig.layout, 1);
            colgap!(fig.layout, 1);
            colgap!(fig.layout, 1);
            
            colgap!(fig.layout, 1);            
            

            fig
            figsave = joinpath(figdir, "$(Int32(it)).png")
            save(figsave, fig)


            # vtkfile = vtk_grid("$loadpath"*"_$(Int32(it+1e4))", xvi[1],xvi[2]) # 2-D VTK file
            # vtkfile["Temperature"] = Array(T_d); vtkfile["MeltFraction"] = Array(ϕ); 
            # vtkfile["Viscosity"] = Array(η_d); vtkfile["Vy"] = Array(Vy_d); vtkfile["Density"] = Array(ρ_d);vtkfile["Density"] = Array(ρ_d);               # Store fields in file
            # outfiles = vtk_save(vtkfile); #=pvd[time/kyr] = vtkfile=# pvd[ustrip.(t_Myrs)] = vtkfile  
        # end

       
        # # if Inject_Dike == true 
        # #     if  ustrip(dimensionalize(t,yr,CharDim)) >= (ustrip(InjectionInterval)*interval) #Inject every time the physical time exceeds the injection interval
        #     if it == 5
        #         ID        =     rand(ind_melt)
        #         cen    =     [x1[ID],z1[ID]]  # Randomly vary center of dike
        #         #   Dike injection based on Toba seismic inversion
        #         if cen[end] < ustrip(nondimensionalize(-25km, CharDim))
        #             Angle_rand = [rand(80.0:0.1:100.0)] # Orientation: near-vertical @ depth             
        #         else
        #             Angle_rand = [rand(-10.0:0.1:10.0)] # Orientation: near-vertical @ shallower depth     
        #         end
                
        #         #   # General center for dike injection
                    
        #         #   if center[end] < ustrip(nondimensionalize(-25km, CharDim))
        #         #     Angle_rand = [rand(80.0:0.1:100.0)] # Orientation: near-vertical @ depth             
        #         #   else
        #         #     Angle_rand = [rand(-10.0:0.1:10.0)] # Orientation: near-vertical @ shallower depth     
        #         #   end
        #         dike      =   Dike(Angle=Angle_rand, W=W_in, H=H_in, Type=Dike_Type, T=T_in, Center=cen, Phase = 2); 
        #         @copy Tnew_cpu Array(@views thermal.T[2:end-1,:])
        #         ## MTK injectDike
        #         Tracers, Tnew_cpu, Vol, dike_poly, Velo  =   MagmaThermoKinematics.InjectDike(Tracers, Tnew_cpu, xvi, dike, nTr_dike);   # Add dike, move hostrocks
        

        #         @parallel assign!((@views thermal.Told[2:end-1,:]), PTArray(Tnew_cpu))
        #         @parallel assign!((@views thermal.T[2:end-1,:]), PTArray(Tnew_cpu)) 
        #         @parallel (@idx size(thermal.Tc)...) temperature2center!(thermal.Tc, thermal.T)

        #         ind = findall(thermal.T .> nondimensionalize(850C, CharDim))
        #         phase_v[ind] .= 2
        #         ind1 = findall(thermal.Tc .> nondimensionalize(800C, CharDim))
        #         phase_c[ind1] .= 2

        #         @parallel (@idx ni) compute_melt_fraction!(ϕ, MatParam, phase_c,  (T=thermal.Tc,))
        #         @parallel (@idx ni) compute_ρg!(ρg[2], ϕ, MatParam,(T=thermal.Tc, P=stokes.P))
        #         @parallel computeViscosity!(η, ϕ, S, mfac, η_f, η_s) #introduces the dike viscosity into the calculation
        #         # copy_arrays_GPU2CPU!(Tnew_cpu, Phi_melt_cpu, thermal.T[2:end-1,:], ϕ)     # Copy arrays to CPU to update properties
        #         UpdateTracers_T_ϕ!(Tracers, xvi, xci, thermal.T, ϕ);      # Update info on tracers 

        #         args = (; ϕ = ϕ,  T = thermal.Tc, P = stokes.P, depth = depth, dt = Inf) # update args
        #         InjectVol +=    Vol
        #             # t_Myrs = dimensionalize(t, Myrs, CharDim)                                                              # Keep track of injected volume
        #             # println("Added new dike; total injected magma volume = $(round(InjectVol/km³,digits=2)) km³; rate Q=$(round(InjectVol/(t_Myrs),digits=2)) m³/s")
        #         println("injected dike; total injected magma volume = $(dimensionalize(InjectVol,km^3,CharDim)); Recharge rate: $(dimensionalize(Vol,km^3,CharDim)/(dimensionalize(dt,yr,CharDim)*ustrip.(InjectionInterval)))")
                
        #     end
        #     if ustrip(dimensionalize(t,yr,CharDim)) > (ustrip(InjectionInterval)*interval)
        #     interval += 1.0 
        #     end
        # end
        push!(evo_t, ustrip(dimensionalize(t,yr,CharDim)))
        push!(evo_InjVol, ustrip(dimensionalize(InjectVol,km^3,CharDim)))
        # if Inject_Dike == true
        #     fig2 = Figure(resolution = (2000, 2000), createmissing = true, fontsize = 40.0)
        #     ax10 = Axis(fig2[1,1], limits = (-0.0, max(ustrip(dimensionalize(t, yr,CharDim)) + 5e2), 0.0, max(InjectVol + 1e2)),xlabel="Time [Myrs]", ylabel="Injected Volume [km³]", title="Injected Volume")
        #     scatter!(ax10, evo_t, evo_InjVol, markersize=40, markeralpha=0.5, markerstrokewidth=0.0, markerstrokecolor=:black, markercolor=:black)
        #     display(fig2)
        # end
        # if minimum(ustrip.(dimensionalize(thermal.T, C, CharDim))) < 0.0
        #     println("Negative temperature")
        #     println(findall(ustrip.(dimensionalize(thermal.T, C, CharDim)) .< 0.0))
        #     break
        # end 
    end
    # Plots.gif(anim, "$gifname"*".gif", fps = 1)
    vtk_save(pvd)

    # finalize_global_grid(; finalize_MPI=true)
    finalize_global_grid()
    print_timer()
    # return (ni=ni, xci=xci, li=li, di=di), thermal
    return thermal, stokes
end

# figdir = "figs2D"
# ni, xci, li, di, figdir ,thermal = DikeInjection_2D();
function run()
    figname = "Temprework_JR_PT"
    nx     = 190
    ny     = 190
    igg    = IGG(init_global_grid(nx, ny, 0)...) 

    DikeInjection_2D(igg; figname=figname,nx=nx, ny=ny);
end

@time run()
