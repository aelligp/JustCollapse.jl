using CUDA
# CUDA.allowscalar(false) # for safety
CUDA.allowscalar(false) # for safety
using JustRelax, JustRelax.DataIO, JustPIC
import JustRelax.@cell

## NOTE: need to run one of the lines below if one wishes to switch from one backend to another
# set_backend("Threads_Float64_2D")
# set_backend("CUDA_Float64_2D")

const USE_GPU=false;
const GPU_ID = 1;

model = if USE_GPU  
    # select if CUDA or AMDGPU
    PS_Setup(:CUDA, Float64, 2)            # initialize parallel stencil in 2D
else        
    PS_Setup(:Threads, Float64, 2)            # initialize parallel stencil in 2D
end
environment!(model)                        

using Printf, Statistics, LinearAlgebra, GeoParams, CairoMakie, SpecialFunctions,  CellArrays    
using ParallelStencil.FiniteDifferences2D   #specify Dimension
using ImplicitGlobalGrid
using MPI: MPI
using GeophysicalModelGenerator, StencilInterpolations, StaticArrays
# using Plots 
using WriteVTK


using MagmaThermoKinematics
using BenchmarkTools
using TimerOutputs
using JLD2

# -----------------------------------------------------------------------------------------
# Viscosity calculation functions stored in this script
include("./src/Particles_Helperfunc.jl");
# include("./src/Helperfunc_old.jl");
include("./src/LoadModel.jl");
println("Loaded helper functions")
#------------------------------------------------------------------------------------------
LonLat                      = load("./Data/ExportToba_2.jld2", "TobaTopography_LATLON");
        
proj                        = ProjectionPoint(; Lat=2.19, Lon=98.91);
Topo                        = Convert2CartData(LonLat, proj);

println("Done loading Model... starting Dike Injection 2D routine")
#------------------------------------------------------------------------------------------

function init_phases!(phases, particles, phases_topo, xc, yc, r)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(phases, px, py, index, phases_topo, xc, yc, r)
        @inbounds for ip in JustRelax.JustRelax.cellaxes(phases)
            # quick escape
            JustRelax.@cell(index[ip, i, j]) == 0 && continue

            x = JustRelax.@cell px[ip, i, j]
            y = -(JustRelax.@cell py[ip, i, j])
            Phase = (phases_topo[i, j])
            if Phase == 1
                @cell phases[ip, i, j] = 1.0

            elseif Phase == 2
                @cell phases[ip, i, j] = 2.0

            elseif Phase == 3
                @cell phases[ip, i, j] = 3.0

            elseif Phase == 4
                @cell phases[ip, i, j] = 4.0

            end

            # # plume - circular
           
            if ((x-xc )^2 + (y + yc)^2 ≤ r^2)
                    JustRelax.@cell phases[ip, i, j] = 2.0

            end


        end
        return nothing
    end

    @parallel (JustRelax.@idx ni) init_phases!(phases, particles.coords..., particles.index,phases_topo, xc , yc, r)
end

# This routine is working and the temp and pressure need to be initialized outside and possibly dimensionalized 

function update_depth!(depth_corrected,phases,topo_interp, x, y)
    nx, ny = length(x), length(y)
  
    for i in 1:nx, j in 1:ny
        
        # vertex coordinates
        xv, yv = x[i], y[j]
        # topography at vertex
        y_topo = topo_interp(xv)
        # depth
        depth = yv

        depth_corrected[i,j] = abs(depth - y_topo)
        if depth > y_topo
            phases[i,j] = 3.0
            # phases[i,j] = Int64(3)
        else 
            # phases[i,j] = Int64(1)
            phases[i,j] = 1.0
        end
    end
    return nothing
end

@parallel_indices (i, j) function init_T!(temperature, phases, depth_corrected, geotherm) 

     if phases[i,j] == 3
        temperature[i+1,j] = 0.0
     else 
        temperature[i+1,j] = geotherm * @all(depth_corrected)
     end
    #  temperature[end-1,j] = temperature[end-3,j]
    #  temperature[end,j] = temperature[end-3,j]
     return nothing 
end

@parallel_indices (i,j) function init_P!(pressure, ρg, phases, depth_corrected)
    if phases[i,j] == 3.0
        @all(pressure) = 0.0
    else
        @all(pressure) = 0.0 + @all(ρg) * @all(depth_corrected)
    end
    return nothing
end

function dirichlet_velocities_pureshear!(Vx, Vy, v_extension, xvi)
    lx = abs(reduce(-, extrema(xvi[1])))
    xv, yv = xvi
    v_extension /= lx/2

    @parallel_indices (i,j) function pure_shear_x!(Vx)
        xi = xv[i] 
        Vx[i, j+1] = v_extension * (xi - lx * 0.5) / 2 
        return nothing
    end

    @parallel_indices (i,j) function pure_shear_y!(Vy)
        yi = abs(yv[j])
        Vy[i+1, j] = v_extension * yi 
        return nothing
    end

    nx, ny = size(Vx)
    @parallel (1:nx, 1:ny-2) pure_shear_x!(Vx)
    nx, ny = size(Vy)
    @parallel (1:nx-2, 1:ny) pure_shear_y!(Vy)

    return nothing
end

function circular_anomaly!(T, anomaly, xc, yc, r, xvi)

    @parallel_indices (i, j) function _circular_anomaly!(T, anomaly, xc, yc, r, x, y)
        @inbounds if (((x[i]-xc )^2 + (y[j] + yc)^2) ≤ r^2)
            T[i, j] = anomaly
            
        end
        return nothing
    end

    ni = length.(xvi)
    @parallel (@idx ni) _circular_anomaly!(T, anomaly, xc, yc, r, xvi...)
    return nothing

end

@parallel_indices (i, j) function compute_melt_fraction!(ϕ, rheology, phase_c, args)
    ϕ[i, j] = compute_meltfraction(rheology, phase_c[i, j], ntuple_idx(args, i, j))
    return nothing
end

# function DikeInjection_2D(igg; figname=figname,nx=nx, ny=ny); 

    #-----------------------------------------------------
    # USER INPUTS
    #-----------------------------------------------------
    Topography= true; #specify if you want topography plotted in the figures
    Freesurface = true; #specify if you want to use freesurface
    Inject_Dike = false; #specify if you want to inject a dike
        dike_width = 2.5km; #specify the width of the dike
        dike_height = 1.0km; #specify the height of the dike
        dike_temp = 1000C; #specify the temperature of the dike  
        InjectionInterval       = 15yr        # number of timesteps between injections (if Inject_Dike=true). Increase this at higher resolution. 
        #now injecting every 2 it
    
    toy = true; #specify if you want to use the toy model or the Toba model
        thermal_perturbation = :circular_anomaly ; #specify if you want a thermal perturbation in the center of the domain or random (:random)
        # thermal_perturbation = :Toba_Inversion ; #specify if you want a thermal perturbation in the center of the domain or random (:random)
        sphere = 1.5km;                               #specify the radius of the circular anomaly
        temp_anomaly = 1000C;
    
    shear = false #specify if you want to use pure shear boundary conditions
        # v_extension = 1e-10m/s
    regime = ViscoElastoPlastic;                  #Rheology of the Stokes Solver: ViscoElastic, ViscoElastoPlastic
    # figname = "testing";              #name of the figure
    
    # nx, ny                  = (190,190).-2  # 2D grid size - should be a multitude of 32 for optimal GPU perfomance 
    Nx,Ny,Nz                = 100,100,100   # 3D grid size (does not yet matter)
        arrow_steps_x_dir         = 7;            # number of arrows in the quiver plot
        arrow_steps_y_dir         = 7;            # number of arrows in the quiver plot

    nt                      = 150             # number of timesteps
   
    η_uppercrust            = 1e21          #viscosity of the upper crust
    η_magma                 = 1e18          #viscosity of the magma
    η_air                   = 1e21          #viscosity of the air
    
    #-----------------------------------------------------

    # IO ------------------------------------------------
    figdir = "./fig2D/$figname/";
    # gifname = "$figname";
    pvd = paraview_collection("./Paraview/$figname");
    !isdir(figdir) && mkpath(figdir);
    # anim = Animation(figdir, String[]);
    # Preparation of VTK/Paraview output 
    # if isdir("./Paraview/$figname/")==false mkdir("./Paraview/$figname/") end; loadpath = "./Paraview/$figname/"; 

    # Standard MTK Toba routine--------------------------------------------------------------

    # nondimensionalize with CharDim 
    CharDim                 = GEO_units(length=40km, viscosity=1e20Pa*s);
    # ni                      = (nx-3,ny-3)   #grid spacing for calculation (without ghost cells)
    ni                      = (nx,ny)   #grid spacing for calculation (without ghost cells)
    # ni                      = (nx,ny,nz)   #grid spacing for calculation (without ghost cells)
    
    Grid                    = CreateCartGrid(size=(Nx,Ny,Nz),x=((Topo.x.val[1,1,1])km,(Topo.x.val[end,1,1])km), y=((Topo.y.val[1,1,1])km,(Topo.y.val[1,end,1])km),z=(-40km,4km));
    X,Y,Z                   = XYZGrid(Grid.coord1D...);
    DataTopo                = CartData(X,Y,Z,(Depthdata=Z,));

    Lon,Lat,Depth           = (Topo.x.val.*km), (Topo.y.val.*km), ((Topo.z.val./1e3).*km);
    Topo_Cart               = CartData(Lon,Lat,Depth,(Depth=Depth,));

    ind                     = AboveSurface(DataTopo,Topo_Cart);
    Phase                   = ones(size(X));
    Phase[ind]             .= 3;

    DataPara                = CartData(X,Y,Z,(Phase=Phase,));
    Phase                   = Int64.(round.(DataPara.fields.Phase));             
      

    # ----------CrossSections for 2D simulation----------------------------------

    # Data_Cross              = CrossSection(DataPara, dims=(ni[1]+1,ni[2]+1), Interpolate=true,Start=(ustrip(-63.5km),ustrip(80.95km)), End=(ustrip(-03.72km), ustrip(20.05km)))
    Data_Cross              = CrossSection(DataPara, dims=(ni[1]+1,ni[2]+1), Interpolate=true,Start=(ustrip(-40.00km),ustrip(50.00km)), End=(ustrip(-06.00km), ustrip(20.00km)));
    # Data_Cross              = CrossSection(DataPara, dims=(ni[1]+1,ni[2]+1), Interpolate=true,Start=(ustrip(-10.00km),ustrip(10.00km)), End=(ustrip(-06.00km), ustrip(00.00km)));
    x_new                   = FlattenCrossSection(Data_Cross);
    Data_Cross              = AddField(Data_Cross,"FlatCrossSection", x_new); 
    Phase                   = dropdims(Data_Cross.fields.Phase,dims=3); #Dropped the 3rd dim of the Phases to be consistent with new grid of the CrossSection
    Phase                   = Int64.(round.(Phase));
    Phase[Phase.==2]  .= 3; #set the phase of the air to 3
    # Topo_Cross              = CrossSection(Topo_Cart, dims=(ni[1]+1,ni[2]+1), Interpolate=true,Start=(ustrip(-63.5km),ustrip(80.95km)), End=(ustrip(-03.72km), ustrip(20.05km)))
    Topo_Cross              = CrossSection(Topo_Cart, dims=(ni[1]+1,ni[2]+1), Interpolate=true,Start=(ustrip(-40.00km),ustrip(50.00km)), End=(ustrip(-06.00km), ustrip(20.00km)));
    # Data_Cross              = CrossSection(DataPara, dims=(ni[1]+1,ni[2]+1), Interpolate=true,Start=(ustrip(-10.00km),ustrip(10.00km)), End=(ustrip(-06.00km), ustrip(00.00km)));
    x_new_Topo                   = FlattenCrossSection(Topo_Cross);
    Topo_Cross              = AddField(Topo_Cross,"FlatCrossSection", x_new_Topo); 
    # Topo_Phase                   = dropdims(Topo_Cross.fields.Topo_Phase,dims=3); #Dropped the 3rd dim of the Phases to be consistent with new grid of the CrossSection
    # Topo_Phase                   = Int64.(round.(Topo_Phase));
    Topo_nondim       = nondimensionalize(Topo_Cross.fields.Depth,CharDim);
    # Topo_2D_nondim    = nondimensionalize(ustrip.(Topo_Cart.fields.Depth),CharDim);
    
    ## Workaround to display topo on heatmaps for η,ρ,ϕ due to different dimensions (ni, not ni+1)
    Data_Cross_ni              = CrossSection(DataPara, dims=(ni[1],ni[2]), Interpolate=true,Start=(ustrip(-40.00km),ustrip(50.00km)), End=(ustrip(-06.00km), ustrip(20.00km)));
    # Data_Cross_ni              = CrossSection(DataPara, dims=(ni[1],ni[2]), Interpolate=true,Start=(ustrip(-10.00km),ustrip(10.00km)), End=(ustrip(-06.00km), ustrip(00.00km)));
    Phase_ni                = dropdims(Data_Cross_ni.fields.Phase,dims=3); #Dropped the 3rd dim of the Phases to be consistent with new grid of the CrossSection
    Phase_ni                = Int64.(round.(Phase_ni));
    Phase_ni[Phase_ni.==2]  .= 3; #set the phase of the air to 3
    #Seismo Model 
    Model3D_Cross              = CrossSection(Model3D_cart, dims=(ni[1]+1,ni[2]+1), Interpolate=true,Start=(ustrip(-40.00km),ustrip(50.00km)), End=(ustrip(-06.00km), ustrip(20.00km)));
    # Model3D_Cross              = CrossSection(Model3D_cart, dims=(ni[1]+3,ni[2]+1), Interpolate=true,Start=(ustrip(-40.00km),ustrip(50.00km)), End=(ustrip(-06.00km), ustrip(20.00km)));
    # Model3D_Cross              = CrossSection(Model3D_cart, dims=(ni[1]+1,ni[2]+1), Interpolate=true,Start=(ustrip(-10.00km),ustrip(10.00km)), End=(ustrip(-06.00km), ustrip(00.00km)));
    Model3D_new             = FlattenCrossSection(Model3D_Cross);
    Model3D_Cross           = AddField(Model3D_Cross,"Model3D_Cross",Model3D_new);

    #seismo model with ni dimensions
    Model3D_Cross_ni              = CrossSection(Model3D_cart, dims=(ni[1],ni[2]), Interpolate=true,Start=(ustrip(-40.00km),ustrip(50.00km)), End=(ustrip(-06.00km), ustrip(20.00km)));
    # Model3D_Cross              = CrossSection(Model3D_cart, dims=(ni[1]+1,ni[2]+1), Interpolate=true,Start=(ustrip(-10.00km),ustrip(10.00km)), End=(ustrip(-06.00km), ustrip(00.00km)));
    Model3D_new_ni            = FlattenCrossSection(Model3D_Cross_ni);
    Model3D_Cross_ni           = AddField(Model3D_Cross_ni,"Model3D_Cross",Model3D_new_ni);
    
    #New 2D Grid
    if Freesurface == true
        Grid2D                  = CreateCartGrid(size=(ni[1],ni[2]), x=(extrema(Data_Cross.fields.FlatCrossSection).*km),z=((minimum(Data_Cross.z.val)./2.66).*km,(maximum(Data_Cross.z.val)).*km), CharDim=CharDim); #create new 2D grid for Injection routine
    else
        # new grid with no topography as the topo now seems to be the problem with strain rate. 
        # Grid2D                  = CreateCartGrid(size=(ni[1],ni[2]), x=(extrema(Data_Cross.fields.FlatCrossSection).*km),z=((minimum(Data_Cross.z.val)./2).*km,0.0.*km), CharDim=CharDim); #create new 2D grid for Injection routine
        Grid2D                  = CreateCartGrid(size=(ni[1],ni[2]), x=(extrema(Data_Cross.fields.FlatCrossSection).*km),z=((-10.0).*km,0.0.*km), CharDim=CharDim); #create new 2D grid for Injection routine
    end

    Phi_melt_data           = dropdims(Model3D_Cross.fields.Phi_melt, dims=3);
    Phi_melt_data_ni           = dropdims(Model3D_Cross_ni.fields.Phi_melt, dims=3);
    
    #Dike location initiation                                                              
    
    ind_melt                =  findall(Phi_melt_data.>0.12); # only Meltfraction of 12% used for dike injection
    ind_melt_ni             =  findall(Phi_melt_data_ni.>0.12); # only Meltfraction of 12% used for dike injection
    x1, z1                  = dropdims(Data_Cross.fields.FlatCrossSection,dims=3), dropdims(Data_Cross.z.val,dims=3); # Define x,z coord for injection
    x1, z1                  = x1.*km, z1.*km; # add dimension to FlatCrossSection to non-dim x,z for consistency
    x1, z1                  = nondimensionalize(x1,CharDim), nondimensionalize(z1,CharDim);

    #-------rheology parameters--------------------------------------------------------------
    # plasticity setup
    do_DP                   = false               # do_DP=false: Von Mises, do_DP=true: Drucker-Prager (friction angle)
    η_reg                   =1.0e16Pas           # regularisation "viscosity" for Drucker-Prager
    τ_y                     = 10MPa              # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    # τ_y                     = Inf              # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ                       = 30.0*do_DP         # friction angle
    G0                      = 1e10Pa             # elastic shear modulus
    Gi                      = G0/(6.0-4.0*do_DP) # elastic shear modulus perturbation
    # Coh                     = 0.0        # cohesion
    Coh                     = τ_y/cosd(ϕ)        # cohesion
    # Coh                     = 15MPa        # cohesion
    εbg                     = 1e-15/s             # background strain rate
    εbg                     = nondimensionalize(εbg, CharDim) # background strain rate
    
    pl                      = DruckerPrager_regularised(C=Coh, ϕ=ϕ, η_vp=η_reg, Ψ=0)        # plasticity
    el                      = SetConstantElasticity(; G=G0, ν=0.47)                            # elastic spring
    el_magma                = SetConstantElasticity(; G=Gi, ν=0.3)                            # elastic spring
    el_air                  = SetConstantElasticity(; ν=0.5, Kb = 0.101MPa)                            # elastic spring

    # disl_creep              = DislocationCreep(A=10^-15.0 , n=2.0, E=476e3, V=0.0  ,  r=0.0, R=8.3145) #AdM    #(;E=187kJ/mol,) Kiss et al. 2023
    disl_creep              = DislocationCreep(A=5.07e-18, n=2.3, E=154e3, V=6e-6,  r=0.0, R=8.3145)
    # diff_creep              = DiffusionCreep()
    creep_rock              = LinearViscous(; η=η_uppercrust*Pa * s)
    creep_magma             = LinearViscous(; η=η_magma*Pa * s)
    creep_air               = LinearViscous(; η=η_air*Pa * s)
    cutoff_visc             = (nondimensionalize(1e16Pa*s,CharDim),nondimensionalize(1e24Pa*s,CharDim))
    β_rock                  = inv(get_Kb(el))
    β_magma                 = inv(get_Kb(el_magma))
    Kb                      = get_Kb(el)

    #-------JustRelax parameters-------------------------------------------------------------
    # Domain setup for JustRelax 
    lx, lz                  = (Grid2D.L[1]), (Grid2D.L[2]) # nondim if CharDim=CharDim
    li                      = lx, lz
    b_width                 = (4, 4, 0) #boundary width
    origin                  = Grid2D.min[1], Grid2D.min[2]
    igg                     = igg
    di                      = @. li / (nx_g(), ny_g()) # grid step in x- and y-direction
    xci, xvi                = lazy_grid(di, li, ni; origin=origin) #non-dim nodes at the center and the vertices of the cell (staggered grid)
    #---------------------------------------------------------------------------------------

    # # Set material parameters                                       
    MatParam                =   (
        #Name="UpperCrust"
        SetMaterialParams(; 
                    Phase   = 1, 
                   Density  = PT_Density(ρ0=2700kg/m^3, β=β_rock/Pa),
               HeatCapacity = ConstantHeatCapacity(cp=1050J/kg/K),
               Conductivity = ConstantConductivity(k=1.5Watt/K/m),       
                 LatentHeat = ConstantLatentHeat(Q_L=350e3J/kg),
          CompositeRheology = CompositeRheology((creep_rock, el, pl, )),
        #   CompositeRheology = CompositeRheology((el, creep_rock, pl, )),
                    Melting = MeltingParam_Caricchi(),
                #  Elasticity = ConstantElasticity(; G=Inf*Pa, Kb=Inf*Pa),
                 Elasticity = el,
                   CharDim  = CharDim,), 

        #Name="Magma" 
        SetMaterialParams(;
                    Phase   = 2, 
                   Density  = PT_Density(ρ0=2250kg/m^3, β=β_magma/Pa),               
               HeatCapacity = ConstantHeatCapacity(cp=1050J/kg/K),
               Conductivity = ConstantConductivity(k=1.5Watt/K/m),       
                 LatentHeat = ConstantLatentHeat(Q_L=350e3J/kg),
          CompositeRheology = CompositeRheology((creep_magma, el_magma)),
                    Melting = MeltingParam_Caricchi(),
                 Elasticity = el_magma,
                   CharDim  = CharDim,),  

        #Name="Sticky Air"
        SetMaterialParams(;  
                    Phase   = 3, 
                  Density   = ConstantDensity(ρ=1000kg/m^3,),                     
               HeatCapacity = ConstantHeatCapacity(cp=1000J/kg/K),
               Conductivity = ConstantConductivity(k=1.5Watt/K/m),       
                 LatentHeat = ConstantLatentHeat(Q_L=0.0J/kg),
          CompositeRheology = CompositeRheology((creep_air,)),
            #    Elasticity = el_air,
                 Elasticity = ConstantElasticity(; G=Inf*Pa, Kb=Inf*Pa),
                    CharDim = CharDim),
                                )  
                        

    #----------------------------------------------------------------------------------

    #update depth with topography !!!CPU ONLY!!!
    phases_topo_v          = zeros(nx+1,ny+1);
    phases_topo_c          = zeros(nx,ny);
    depth_corrected_v      = zeros(nx+1,ny+1);
    depth_corrected_c      = zeros(nx,ny);

    topo_interp = linear_interpolation(xvi[1], Topo_nondim);
    update_depth!(depth_corrected_v, phases_topo_v,topo_interp, xvi...);
    
    depth_corrected_v = PTArray(depth_corrected_v);
    depth_corrected_c = PTArray(depth_corrected_c);
    phases_topo_v     = (PTArray(phases_topo_v));
    phases_topo_c     = (PTArray(phases_topo_c));
    @parallel vertex2center!(phases_topo_c,phases_topo_v);
    @parallel vertex2center!(depth_corrected_c,depth_corrected_v);
    
    @views phases_topo_c = round.(phases_topo_c);
    @views phases_topo_c[phases_topo_c .== 2.0] .= 3.0;
    
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 20, 40, 1 # nxcell = initial number of particles per cell; max_cell = maximum particles accepted in a cell; min_xcell = minimum number of particles in a cell
    particles = init_particles_cellarrays(
        nxcell, max_xcell, min_xcell, xvi[1], xvi[2], di[1], di[2], ni[1], ni[2]
    );
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases      = init_particle_fields_cellarrays(particles, Val(3));
    particle_args    = (pT, pPhases);


    xc, yc      = 0.66*lx, 0.4*lz - abs(minimum((xvi[2])))  # origin of thermal anomaly
    radius           = nondimensionalize(sphere, CharDim)         # radius of perturbation
    init_phases!(pPhases, particles, phases_topo_v, xc,yc, radius);
    phase_ratios     = PhaseRatio(ni, length(MatParam));
    @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases);

    #not yet nice
    p        = particles.coords;
    pp = [argmax(p) for p in phase_ratios.center] #if you want to plot it in a heatmap rather than scatter


    # Physical Parameters 
    # ΔT                      =   nondimensionalize(450C, CharDim)
    # GeoT                    =   -(ΔT - nondimensionalize(0C, CharDim)) / li[2]
    geotherm                = 30 # C/km

    η                       =   MatParam[2].CompositeRheology[1][1].η.val       # viscosity for the Rayleigh number
    # η                       =   MatParam[2].CompositeRheology[1][2].η.val       # viscosity for the Rayleigh number
    Cp0                      =   MatParam[2].HeatCapacity[1].cp.val              # heat capacity     
    ρ0                      =   MatParam[2].Density[1].ρ0.val                   # reference Density
    k0                      =   MatParam[2].Conductivity[1]              # Conductivity
    G                       =   MatParam[1].Elasticity[1].G.val                 # Shear Modulus
    κ                       =   nondimensionalize(1.5Watt/K/m,CharDim)/(ρ0 * Cp0);                                   # thermal diffusivity
    g                       =   MatParam[1].Gravity[1].g.val                    # Gravity
    # α                       =   0.03
    α                       =   MatParam[1].Density[1].α.val                    # thermal expansion coefficient for PT Density
    # Ra                      =   ρ0 * g * α * ΔT * 10^3 / (η * κ)                # Rayleigh number
    dt                      = dt_diff = 0.5 * min(di...)^2 / κ / 2.01           # diffusive CFL timestep limiter

    v_extension             = nondimensionalize(1e-10m/s, CharDim)   # extension velocity for pure shear boundary conditions
    relaxation_time         = nondimensionalize(η_magma  *Pas, CharDim)/nondimensionalize(Gi, CharDim) # η_magma/Gi
    # Initialize arrays for PT thermal solver
    k                       = @fill(nondimensionalize(1.5Watt/K/m,CharDim), ni...);
    ρCp                     = @fill(ρ0.*Cp0, ni...);
    # CFL_thermal             = (κ * (2*dt/(max.(di...)^2)))./2 # CFL number for thermal diffusion - not in use right now

    # Initialisation 
    thermal                 = ThermalArrays(ni);                                # initialise thermal arrays and boundary conditions
    thermal_bc              = TemperatureBoundaryConditions(; 
                no_flux     = (left = true , right = true , top = false, bot = false), 
                periodicity = (left = false, right = false, top = false, bot = false),
    );

    depth_corrected_dim    = ustrip.(dimensionalize(depth_corrected_v,km,CharDim))
    @parallel (@idx ni .+ 1) init_T!(thermal.T, phases_topo_v, depth_corrected_dim,geotherm)
    thermal.T             .=nondimensionalize((thermal.T).*C,CharDim)
    thermal_bcs!(thermal.T, thermal_bc)

    @parallel (JustRelax.@idx size(thermal.Tc)...) temperature2center!(thermal.Tc, thermal.T)
    
    Tnew_cpu                = Array{Float64}(undef, ni.+1...);                  # Temperature for the CPU
    Phi_melt_cpu            = Array{Float64}(undef, ni...);                    # Melt fraction for the CPU
    
    stokes                  = StokesArrays(ni, ViscoElastic);                         # initialise stokes arrays with the defined regime
    pt_stokes               = PTStokesCoeffs(li, di; ϵ=1e-5,  CFL=0.8 / √2.1); #ϵ=1e-4,  CFL=1 / √2.1 CFL=0.27 / √2.1

    args                    = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    # pt_thermal = PTThermalCoeffs(k, ρCp, dt, di, li; CFL=CFL_thermal)
    # pt_thermal = PTThermalCoeffs(k, ρCp, dt, di, li)
    pt_thermal       = PTThermalCoeffs(
        MatParam, phase_ratios, args, dt, ni, di, li; ϵ=1e-5, CFL=1e-3 / √3
    )
    # Boundary conditions of the flow
    if shear == true
        dirichlet_velocities_pureshear!(@velocity(stokes)..., v_extension, xvi);
        # pureshear_bc!(stokes, xci, xvi, εbg)
    end
    flow_bcs                = FlowBoundaryConditions(; 
                free_slip   = (left=true, right=true, top=true, bot=true), 
    );
   
   
    # phase_v                 = Int64.(PTArray(ones(ni.+1...)));                 
    # phase_v                 = Int64.(PTArray(ones(ni[1].+3,ni[2].+1)));         # phase array for the vertices with dim of thermal.T
    # phase_c                 = Int64.(PTArray(ones(ni...)));                     # phase array for the center
    η                       = @ones(ni...);                                     # initialise viscosity       
    η_vep                   = deepcopy(η);                                       # initialise viscosity for the VEP
    G                       = @fill(MatParam[1].Elasticity[1].G.val, ni...);     # initialise shear modulus
    ϕ                       = similar(η);                                        # melt fraction center
    ϕ_v                     = @zeros(ni.+1...);                                  # melt fraction vertices
    S, mfac                 = 1.0, -2.8;                                         # factors for hexagons (remnants of the old code, but still used in functions) Deubelbeiss, Kaus, Connolly (2010) EPSL   
    # η_f                     = MatParam[2].CompositeRheology[1][1].η.val         # melt viscosity
    # # η_f                     = nondimensionalize(1e18Pas,CharDim )         # melt viscosity
    # η_s                     = MatParam[1].CompositeRheology[1][1].η.val         # solid viscosity
    # # η_s                     = nondimensionalize(1e23Pas, CharDim)         # solid viscosity

    # Buoyancy force
    ρg                      = @zeros(ni...), @zeros(ni...)                      # ρg[1] is the buoyancy force in the x direction, ρg[2] is the buoyancy force in the y direction                                   
     
    P_init    = @zeros(ni...);
    P_dike    = @zeros(ni...);
    P_chamber = @zeros(ni...);
    Qmask     = @zeros(ni...);
    Qmass     = 0.15;
    P_Dirichlet = @zeros(ni...);
    SH        = @zeros(ni...); #shear heating to be Updated
    # Preparation for Visualisation

    ni_v_viz = nx_v_viz, ny_v_viz = (ni[1]-1)*igg.dims[1], (ni[2]-1)*igg.dims[2];      # size of the visualisation grid on the vertices according to MPI dims
    ni_viz = nx_viz, ny_viz = (ni[1]-2)*igg.dims[1], (ni[2]-2)*igg.dims[2];            # size of the visualisation grid on the vertices according to MPI dims
    # ni_viz = nx_viz, ny_viz = (ni[1])*igg.dims[1], (ni[2])*igg.dims[2];            # size of the visualisation grid on the vertices according to MPI dims
    Vx_vertex               = PTArray(ones(ni.+1...));                                                  # initialise velocity for the vertices in x direction
    Vy_vertex               = PTArray(ones(ni.+1...));                                                  # initialise velocity for the vertices in y direction

    # Arrays for visualisation
    Tc_viz                  = Array{Float64}(zeros(ni_viz...));                                   # Temp center with ni
    Vx_viz                  = Array{Float64}(zeros(ni_v_viz...));                                 # Velocity in x direction with ni_viz .-1
    Vy_viz                  = Array{Float64}(zeros(ni_v_viz...));                                 # Velocity in y direction with ni_viz .-1
    ∇V_viz                  = Array{Float64}(zeros(ni_viz...));                                   # Velocity in y direction with ni_viz .-1
    P_viz                   = Array{Float64}(zeros(ni_viz...));                                   # Pressure with ni_viz .-2
    τxy_viz                 = Array{Float64}(zeros(ni_v_viz...));                                 # Shear stress with ni_viz .-1
    τII_viz                 = Array{Float64}(zeros(ni_viz...));                                   # 2nd invariant of the stress tensor with ni_viz .-2
    εII_viz                 = Array{Float64}(zeros(ni_viz...));                                   # 2nd invariant of the strain tensor with ni_viz .-2
    εxy_viz                 = Array{Float64}(zeros(ni_v_viz...));                                 # Shear strain with ni_viz .-1
    η_viz                   = Array{Float64}(zeros(ni_viz...));                                   # Viscosity with ni_viz .-2 
    η_vep_viz               = Array{Float64}(zeros(ni_viz...));                                   # Viscosity for the VEP with ni_viz .-2
    ϕ_viz                   = Array{Float64}(zeros(ni_viz...));                                   # Melt fraction with ni_viz .-2
    ρg_viz                  = Array{Float64}(zeros(ni_viz...));                                   # Buoyancy force with ni_viz .-2
    phase_v_viz             = Array{Int64}(zeros(ni_v_viz...));                                   # Phase array with ni_viz .-1
    phase_c_viz             = Array{Int64}(zeros(ni_viz...));                                     # Phase array with ni_viz .-2
    # Arrays for gathering data from MPI blocks
    T_inn                   = PTArray(zeros(ni_viz...));
    Vx_inn                  = PTArray(zeros(ni_v_viz...));
    Vy_inn                  = PTArray(zeros(ni_v_viz...));
    ∇V_inn                  = PTArray(zeros(ni_viz...));
    P_inn                   = PTArray(zeros(ni_viz...)) ;
    τxy_inn                 = PTArray(zeros(ni_v_viz...));
    τII_inn                 = PTArray(zeros(ni_viz...));
    εII_inn                 = PTArray(zeros(ni_viz...));
    εxy_inn                 = PTArray(zeros(ni_v_viz...));
    η_inn                   = PTArray(zeros(ni_viz...));
    η_vep_inn               = PTArray(zeros(ni_viz...));
    ϕ_inn                   = PTArray(zeros(ni_viz...));
    ρg_inn                  = PTArray(zeros(ni_viz...));
    phase_v_inn             = Int64.(PTArray(zeros(ni_v_viz...)));
    phase_c_inn             = Int64.(PTArray(zeros(ni_viz...)));


    # Arguments for functions
    args                    = (; ϕ = ϕ,  T = thermal.Tc, P = stokes.P, dt = Inf)#, S=S, mfac=mfac, η_f=η_f, η_s=η_s) # this is used for various functions, remember to update it within loops.

    # Dike parameters 
    W_in, H_in              = nondimensionalize(dike_width, CharDim),nondimensionalize(dike_height, CharDim); # Width and thickness of dike
    T_in                    = nondimensionalize(dike_temp, CharDim);  
    # H_ran, W_ran            = length(z1) ,length(x1);                          # Size of domain in which we randomly place dikes and range of angles
    # H_ran, W_ran            = (xvi[1].stop, xvi[2].start).* [0.3;0.4];                          # Size of domain in which we randomly place dikes and range of angles
    H_ran, W_ran            = (xvi[1].stop, xvi[2].start).* [0.2;0.3];                          # Size of domain in which we randomly place dikes and range of angles
    Dike_Type               = "ElasticDike"
    Tracers                 = StructArray{Tracer}(undef, 1);
    nTr_dike                = 300; 
   
    # # -Topography -------------------------------------------------------
    # if Topography == true && Freesurface == true
    #     #update phase on the vertices
    #     @views phase_v[Phase .== 3] .= 3
    #     #update phase on the center
    #     @views phase_c[Phase_ni .== 3] .= 3       
    # elseif Topography == false && Freesurface == true
    #     let
    #         xci_phase, xvi_phase = lazy_grid(di, li, (ni[1].+2,ni[2]); origin=origin)  
    #         Yv = [y for x in xvi_phase[1], y in xvi_phase[2]]
    #         Yc = [y for x in xci[1], y in xci[2]]
    #         mask = Yv .> 0
    #         @views phase_v[mask] .= 3
    #         mask = Yc .> 0
    #         @views phase_c[mask] .= 3
    #     end
    # end
   


    
   
    @copy  thermal.Told thermal.T;
    @copy  Tnew_cpu Array(thermal.T[2:end-1,:]);


    
    args                    = (; ϕ = ϕ,  T = thermal.Tc, P = stokes.P, dt = Inf)#, S=S, mfac=mfac, η_f=η_f, η_s=η_s) # this is used for various functions, remember to update it within loops.
    
    for _ in 1:2
        @parallel (JustRelax.@idx ni) compute_ρg!(ρg[2], phase_ratios.center, MatParam, (T=thermal.Tc, P=stokes.P))
        @parallel (@idx ni) init_P!(stokes.P, ρg[2], phases_topo_c, depth_corrected_c)
    end 
    
    @parallel (@idx ni) compute_viscosity!(
        η, 1.0, phase_ratios.center, @strain(stokes)..., args, MatParam, cutoff_visc
        )
        η_vep            = copy(η)
        
    @parallel (@idx ni) compute_melt_fraction!(args.ϕ, MatParam, phase_ratios.center,  (T=thermal.Tc,))

        if Freesurface == true
            xc, yc      = 0.66*lx, 0.4*lz #- abs(minimum((xvi[2])))  # origin of thermal anomaly
        else 
            # xc, yc      = 0.66*lx, 0.5*lz  # origin of thermal anomaly
            xc, yc      = 0.5*lx, 0.5*lz   # origin of thermal anomaly
        end
        if toy == true
            if thermal_perturbation == :random
                δT          = 5.0              # thermal perturbation (in %)
                random_perturbation!(thermal.T, δT, (lx*1/8, lx*7/8), (-2000e3, -2600e3), xvi)
                @show "random perturbation"
            elseif thermal_perturbation == :circular
                δT          = 10.0              # thermal perturbation (in %)
                r           = nondimensionalize(5km, CharDim)         # radius of perturbation
                circular_perturbation!(thermal.T, δT, xc, yc, r, xvi)
                @show "circular perturbation"
            elseif thermal_perturbation == :circular_anomaly
                anomaly     = nondimensionalize(temp_anomaly,CharDim) # temperature anomaly
                # anomaly     = ustrip.(temp_anomaly) # temperature anomaly
                radius           = nondimensionalize(sphere, CharDim)         # radius of perturbation
                circular_anomaly!(thermal.T, anomaly, xc, yc, radius, xvi)
                @show "circular anomaly"
                
                ΔP = nondimensionalize(1MPa, CharDim) 
                @views stokes.P[phase_ratios.center .== 2.0] .+= ΔP 
                @info "Pressure anomaly added with an overpressure of $(dimensionalize(ΔP,MPa, CharDim))"
            # elseif thermal_perturbation == :Toba_Inversion
            #     # Toba Inversion
            #     @views phase_c[Phi_melt_data_ni.>0.12]  .=2
            #     @views phase_v[Phi_melt_data.>0.12]  .=2
    
            #     anomaly     = nondimensionalize(temp_anomaly,CharDim) # temperature anomaly
            #     @views thermal.T[Phi_melt_data.>0.12] .= anomaly
            
                # ΔP = nondimensionalize(1MPa, CharDim)
                # @views stokes.P[phase_c .== 2] .+= ΔP 
                # @info "Pressure anomaly added with an overpressure of $(dimensionalize(ΔP,MPa, CharDim))"
    
    
            end
        end
    
        pxc = PTArray([x for x in xci[1], y in xci[2]]); # x-coordinates of the center for Pressure Gaussian
        pyc = PTArray([y for x in xci[1], y in xci[2]]); # y-coordinates of the center for Pressure Gaussian
    
        # if toy == true
        #     xc, yc      = 0.68*lx, 0.15*lz  # origin of thermal anomaly
        #     #Pressure Gaussian 
        #     Qmask .= 0.15.* exp.(.-128.0((pxc.-xc).^2.0 + (pyc.+yc).^2.0));
        #     # Qmask .= Qmass.* exp.(.-128.0((pxc.-xc).^2.0 + (pyc.+yc).^2.0));
        #     # @views P_chamber[phase_c .==2 ] .= Qmask[phase_c .== 2]; 
        #     @views P_chamber[phase_c .== 2] .= nondimensionalize(1MPa,CharDim)
        # end
        # @parallel (@idx ni) computeViscosity!(η, ϕ, S, mfac, η_f, η_s)
        # ν = 0.05
        #     @parallel (@idx ni) compute_viscosity_MTK!(
        #         η, ν, @strain(stokes)..., args, MatParam, phase_c
        #     )

    # ----------------------------------------------------  
   
    # Time loop
    t, it       = 0.0, 0
    interval    = 1.0
    InjectVol   = 0.0;
    evo_t       = Float64[]
    evo_InjVol  = Float64[]
    local iters             

    T_buffer    = @zeros(ni.+1) 
    Told_buffer = similar(T_buffer)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end
    JustPIC.grid2particle!(pT, xvi, T_buffer, particles.coords)

    # @parallel (@idx ni) update_G!(G, MatParam, phase_ratios.center);
    @copy stokes.P0 stokes.P;
    P_init .= stokes.P;
    # ##testing pressure with nils code
    # maskrho_solid_and_chamber = BitArray{2}(undef, ni...)
    # @views maskrho_solid_and_chamber[phase_c .==3] .= 0  
    # @views maskrho_solid_and_chamber[phase_c .<3] .= 1
    # stokes.P .= reverse(cumsum(ρg[2].*reverse(maskrho_solid_and_chamber,dims=2),dims=2),dims=2)
    # stokes.P .= stokes.P .*maskrho_solid_and_chamber
    # # P .= cumsum(ρg[2].*(phase_c, dims=2),dims=2)

     # Plot initial T and η profiles
     let
        Yv  = [y for x in xvi[1], y in xvi[2]][:]
        Y   = [y for x in xci[1], y in xci[2]][:]
        fig = Figure(resolution = (1200, 900))
        ax1 = Axis(fig[1,1], aspect = 2/3, title = "T")
        ax2 = Axis(fig[1,2], aspect = 2/3, title = "Pressure")
        lines!(ax1, Array(ustrip.(dimensionalize(thermal.T[2:end-1,:][:],C,CharDim))), ustrip.(dimensionalize(Yv,km,CharDim)))
        lines!(ax2, Array(ustrip.(dimensionalize(stokes.P[:],MPa,CharDim))), ustrip.(dimensionalize(Y,km,CharDim)))
        # ylims!(ax1, minimum(xvi[2]), 0)
        # ylims!(ax2, minimum(xvi[2]), 0)
        hideydecorations!(ax2)
        save(joinpath(figdir, "initial_profile.png"), fig)
        fig
    end

    

    # while it < nt   
     
        args = (; ϕ = ϕ,  T = thermal.Tc, P = stokes.P, dt = Inf)#, S=S, mfac=mfac, η_f=η_f, η_s=η_s); 

        @parallel (@idx ni) compute_viscosity!(
        η, 1.0, phase_ratios.center, @strain(stokes)..., args, MatParam, cutoff_visc
        )
        @parallel (JustRelax.@idx ni) compute_ρg!(ρg[2], phase_ratios.center, MatParam, args)

        if Inject_Dike == true
              if rem(it, 2) == 0
            # if  ustrip(dimensionalize(t,yr,CharDim)) >= (ustrip(InjectionInterval)*interval) #Inject every time the physical time exceeds the injection interval
                ID             = rand(ind_melt)
                cen            = [x1[ID],z1[ID]]  # Randomly vary center of dike
                # cen              = ((xvi[1].stop, xvi[2].stop).+(xvi[1].start,xvi[2].start))./2.0 .+ rand(-0.5:1e-3:0.5, 2).*[W_ran;H_ran]; 
                #   Dike injection based on Toba seismic inversion
                if cen[end] < ustrip(nondimensionalize(-5km, CharDim))
                    Angle_rand = [rand(80.0:0.1:100.0)] # Orientation: near-vertical @ depth             
                else
                    Angle_rand = [rand(-10.0:0.1:10.0)] # Orientation: near-vertical @ shallower depth     
                end
                
                dike      =   Dike(Angle=Angle_rand, W=W_in, H=H_in, Type=Dike_Type, T=T_in, Center=cen, #=ΔP=10e6,=#Phase = 2);
                @copy Tnew_cpu Array(@views thermal.T[2:end-1,:])
                ## MTK injectDike
                Tracers, Tnew_cpu, Vol, dike_poly, Velo  =   MagmaThermoKinematics.InjectDike(Tracers, Tnew_cpu, xvi, dike, nTr_dike);   # Add dike, move hostrocks
                                
                @views thermal.Told[2:end-1,:] .= PTArray(Tnew_cpu)
                @views thermal.T[2:end-1,:] .= PTArray(Tnew_cpu)
                
                copy_arrays_GPU2CPU!(Tnew_cpu, Phi_melt_cpu, thermal.T[2:end-1,:], ϕ)     # Copy arrays to CPU to update properties
                UpdateTracers_T_ϕ!(Tracers, xvi, xci, Tnew_cpu, Phi_melt_cpu);      # Update info on tracers 
        
                @parallel (@idx size(thermal.Tc)...) temperature2center!(thermal.Tc, thermal.T)
                args = (; ϕ = ϕ,  T = thermal.Tc, P = stokes.P, dt = Inf, S=S, mfac=mfac, η_f=η_f, η_s=η_s) 
                    #   add flux to injection
                InjectVol +=    Vol
                    # t_Myrs = dimensionalize(t, Myrs, CharDim)                                                              # Keep track of injected volume
                    # println("Added new dike; total injected magma volume = $(round(InjectVol/km³,digits=2)) km³; rate Q=$(round(InjectVol/(t_Myrs),digits=2)) m³/s")
                println("injected dike; total injected magma volume = $(dimensionalize(InjectVol,km^3,CharDim)); Recharge rate: $(dimensionalize(Vol,km^3,CharDim)/(dimensionalize(dt,yr,CharDim)*InjectionInterval))")
                
            end
        end
        # P_dike = PTArray(zeros(ni...))
        P_dike .*= exp(-t/relaxation_time);
        P_chamber .*= exp(-t/relaxation_time);
        # @views P_dike[thermal.Tc .== nondimensionalize(dike_temp, CharDim)] .= nondimensionalize(1MPa, CharDim)
        if ustrip(dimensionalize(t,yr,CharDim)) > (ustrip(InjectionInterval)*interval)
            interval += 1.0 
        end
        
        # if maximum(stokes.P.-P_init) >= nondimensionalize(25MPa, CharDim)
        #     Qmask .= 0.0.*Qmask
        #     println("INJECTION STOPPED")
        #     # @views stokes.P[phase_c .== 2] .= stokes.P0[phase_c .== 2] #.+ nondimensionalize(5MPa,CharDim)
        # end
        # if it == 8 
        #     Inject_Dike = false 
        #     shear = true
        #     if shear == true
        #         dirichlet_velocities_pureshear!(@velocity(stokes)..., v_extension, xvi);
        #         # pureshear_bc!(stokes, xci, xvi, εbg)
        #     end
        #     flow_bcs                = FlowBoundaryConditions(; 
        #                 free_slip   = (left=true, right=true, top=true, bot=true), 
        #     );        

        # end
        if Inject_Dike == true 
            @views P_Dirichlet .= stokes.P .+ P_dike
        else
            if it == 1
             @views P_Dirichlet = stokes.P .+ P_chamber #P_Dike is the gaussian pressure gradient in this case
            else 
             @views P_Dirichlet = stokes.P #not pressurizing chamber
            end
            # @views P_Dirichlet = stokes.P #not pressurizing chamber
        end 
        if shear == true 
                @views P_Dirichlet[ϕ .> 0.12] = stokes.P[ϕ .> 0.12] + P_chamber[ϕ .> 0.12]
        end

        @parallel (@idx ni) compute_ρg!(ρg[2], phase_ratios.center, MatParam,(T=thermal.Tc, P=stokes.P))
        @copy stokes.P0 stokes.P;
        args = (; ϕ = ϕ,  T = thermal.Tc, P = stokes.P, dt = Inf)#, S=S, mfac=mfac, η_f=η_f, η_s=η_s) 
        # Stokes solver ----------------
        # iters = MTK_solve!(
        #     stokes,
        #     pt_stokes,
        #     di,
        #     flow_bcs,
        #     ρg,
        #     η, 
        #     η_vep,
        #     phase_c,
        #     P_Dirichlet,
        #     args,
        #     MatParam, # do a few initial time-steps without plasticity to improve convergence
        #     dt,   # if no elasticity then Inf otherwise dt
        #     igg;
        #     iterMax=500e3,  # 10e3 for testing
        #     nout=5000,
        #     b_width,
        #     verbose=true,
        # )
        solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            η,
            η_vep,
            phase_ratios,
            MatParam,
            args,
            dt,
            igg;
            iterMax=50e3,
            nout=1e3,
            b_width,
            viscosity_cutoff=cutoff_visc
        )
        @parallel (JustRelax.@idx ni) tensor_invariant!(stokes.ε.II, @strain(stokes)...)
          dt = compute_dt(stokes, di, dt_diff,igg)
        # ------------------------------
       @show dt

        # interpolate fields from particle to grid vertices
        particle2grid!(T_buffer, pT, xvi, particles.coords)
        @views T_buffer[:, end]      .= nondimensionalize(0.0C,CharDim)
        @views thermal.T[2:end-1, :] .= T_buffer
        temperature2center!(thermal)


        # Thermal solver ---------------
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            MatParam,
            args,
            dt,
            di;
            igg     = igg,
            phase   = phase_ratios,
            iterMax = 10e3,
            nout    = 1e2,
            verbose = true,
        )
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection_RK!(particles, @velocity(stokes), grid_vx, grid_vy, dt, 2 / 3)
        # advect particles in memory
        shuffle_particles!(particles, xvi, particle_args)        
        # interpolate fields from grid vertices to particles
        for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
            copyinn_x!(dst, src)
        end
        grid2particle_flip!(pT, xvi, T_buffer, Told_buffer, particles.coords)
        # check if we need to inject particles
        inject = check_injection(particles)
        inject && inject_particles_phase!(particles, pPhases, (pT, ), (T_buffer,), xvi)
        # update phase ratios
        @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases)
     
     
        # Update buoyancy and viscosity -
        @copy thermal.Told thermal.T
        @parallel (@idx size(thermal.Tc)...) temperature2center!(thermal.Tc, thermal.T)

        # Update buoyancy and viscosity -
        @copy thermal.Told thermal.T
        @parallel (@idx ni) compute_melt_fraction!(ϕ, MatParam, phase_ratios.center,  (T=thermal.Tc,))
        @parallel center2vertex!(ϕ_v,ϕ)
        # @parallel update_phase(phase_c, ϕ)
        # @parallel update_phase_v(phase_v, ϕ_v)
        # @parallel (@idx ni) compute_ρg!(ρg[2], args.ϕ, MatParam, (T=args.T, P=args.P))
        # @parallel (@idx ni) compute_ρg_phase!(ρg[2], phase_c, MatParam, (T=args.T, P=args.P))
        # @parallel (@idx ni) computeViscosity!(η, ϕ, S, mfac, η_f, η_s)

        # @parallel (@idx ni) tensor_invariant!(stokes.ε.II, @strain(stokes)...)
        # stokes.τ.II[phase_ratios.center .== 3] .= nondimensionalize(0.0MPa, CharDim)
        # @copy η_vep η

        @show it += 1
        t += dt
        if it == 1 @show Ra end

        #  # # Plotting -------------------------------------------------------
        # if it == 1 || rem(it, 1) == 0
            checkpointing(figdir, stokes, thermal.T, η, t)
            # Arrow plotting routine
            Xc, Yc    = [x for x=xvi[1], y=xvi[2]], [y for x=xvi[1], y=xvi[2]]
            st_x        = arrow_steps_x_dir                                       # quiver plotting spatial step x direction
            st_y        = arrow_steps_y_dir                                       # quiver plotting spatial step y direction
            Xp, Yp    = Xc[1:st_x:end,1:st_y:end], Yc[1:st_x:end,1:st_y:end]
            Vxp       = Array(0.5*(stokes.V.Vx[1:st_x:end-1,1:st_y:end  ] + stokes.V.Vx[2:st_x:end,1:st_y:end]))
            Vyp       = Array(0.5*(stokes.V.Vy[1:st_x:end  ,1:st_y:end-1] + stokes.V.Vy[1:st_x:end,2:st_y:end]))
            # Vscale    = 0.5/maximum(sqrt.(Vxp.^2 + Vyp.^2)) * di[1]*(st_x-1)

            JustRelax.velocity2vertex!(Vx_vertex, Vy_vertex, stokes.V.Vx,stokes.V.Vy)
            Xp_d = Array(ustrip.(dimensionalize(Xp, km, CharDim)));
            Yp_d = Array(ustrip.(dimensionalize(Yp, km, CharDim)));

            x_v = ustrip.(dimensionalize(xvi[1][2:end-1], km, CharDim))  #not sure about this with MPI and the size (intuition says should be fine)
            y_v = ustrip.(dimensionalize(xvi[2][2:end-1], km, CharDim))
            x_c = ustrip.(dimensionalize(xci[1][2:end-1], km, CharDim))
            y_c = ustrip.(dimensionalize(xci[2][2:end-1], km, CharDim))

            T_inn = Array(thermal.Tc[2:end-1,2:end-1]);
            Vx_inn = Array(Vx_vertex[2:end-1,2:end-1]);
            Vy_inn = Array(Vy_vertex[2:end-1,2:end-1]);
            ∇V_inn = Array(stokes.∇V[2:end-1,2:end-1]);
            P_inn = Array(stokes.P[2:end-1,2:end-1]);
            τII_inn = Array(stokes.τ.II[2:end-1,2:end-1]);
            τxy_inn = Array(stokes.τ.xy[2:end-1,2:end-1]);
            εII_inn = Array(stokes.ε.II[2:end-1,2:end-1]);
            εxy_inn = Array(stokes.ε.xy[2:end-1,2:end-1]);
            η_inn = Array(η[2:end-1,2:end-1]);
            η_vep_inn = Array(η_vep[2:end-1,2:end-1]);
            ϕ_inn = Array(ϕ[2:end-1,2:end-1]);
            ρg_inn = Array(ρg[2][2:end-1,2:end-1]);
            # phase_v_inn = Array(phase_v[3:end-2,2:end-1]);
            # phase_c_inn = Array(phase_c[2:end-1,2:end-1]);

            gather!(T_inn, Tc_viz)
            gather!(Vx_inn, Vx_viz)
            gather!(Vy_inn, Vy_viz)
            gather!(∇V_inn, ∇V_viz)
            gather!(P_inn, P_viz)
            gather!(τII_inn, τII_viz)
            gather!(τxy_inn, τxy_viz)
            gather!(εII_inn, εII_viz)
            gather!(εxy_inn, εxy_viz)
            gather!(η_inn, η_viz)
            gather!(η_vep_inn, η_vep_viz)
            gather!(ϕ_inn, ϕ_viz)
            gather!(ρg_inn, ρg_viz)
            # gather!(phase_v_inn, phase_v_viz)
            # gather!(phase_c_inn, phase_c_viz)

        
            T_d = ustrip.(dimensionalize(Array(Tc_viz), C, CharDim))
            η_d = ustrip.(dimensionalize(Array(η_viz), Pas, CharDim))
            η_vep_d = ustrip.(dimensionalize(Array(η_vep_viz), Pas, CharDim))
            Vy_d= ustrip.(dimensionalize(Array(Vy_viz),   cm/yr, CharDim));
            Vx_d= ustrip.(dimensionalize(Array(Vx_viz),   cm/yr, CharDim));
            ∇V_d= ustrip.(dimensionalize(Array(∇V_viz),   cm/yr, CharDim));
            P_d = ustrip.(dimensionalize(Array(P_viz),   MPa, CharDim));
            # Velo_d= ustrip.(dimensionalize(Array(Vy_updated),   cm/yr, CharDim));
            ρg_d= ustrip.(dimensionalize(Array(ρg_viz),   kg/m^3*m/s^2, CharDim))
            ρ_d = ρg_d/10;
            ϕ_d = Array(ϕ_viz)
            τII_d = ustrip.(dimensionalize(Array(τII_viz),   MPa, CharDim))
            τxy_d = ustrip.(dimensionalize(Array(τxy_viz),   MPa, CharDim))
            # τxx_d = ustrip.(dimensionalize(Array(stokes.τ.xx),   MPa, CharDim))
            εII_d = ustrip.(dimensionalize(Array(εII_viz),   s^-1, CharDim))
            εxy_d = ustrip.(dimensionalize(Array(εxy_viz),   s^-1, CharDim))
            t_yrs = dimensionalize(t, yr, CharDim)
            t_Kyrs = t_yrs/1e3
            t_Myrs = t_Kyrs/1e3
            
            
            ind_topo = findall(pp.==3)
            ind_ni  = findall(pp[2:end-1,2:end-1].==3)
            # ind_ni  = findall(phases_new.==3)

            # if Topography == true && Freesurface == true
            #     T_d[ind_ni].= NaN                 
            #     Vy_d[ind_topo].=NaN                 
            #     Vx_d[ind_topo].=NaN                 
            #     η_d[ind_ni].= NaN                   
            #     η_vep_d[ind_ni].= NaN                   
            #     ϕ_d[ind_ni]  .= NaN                   
            #     # τII_d[ind_ni].= NaN
            #     P_d[ind_ni].= NaN
            #     ρ_d[ind_ni].= NaN 
            # #     εxy_d[ind_ni].= NaN
            # #     εII_d[ind_ni].= NaN
            # elseif Topography == false && Freesurface == true
            #     ρ_d[ind_ni].= NaN # dont comment bc free surface is too low to see magma difference properly
            #     # εxy_d[ind_ni].= NaN
            # end
            if igg.me == 0
                fig = Figure(resolution = (2000, 2000), createmissing = true)
                ar = li[1]/li[2]
                # ar = DataAspect()

                ax0 = Axis(fig[1,1:2], aspect = ar,title="t = $((ustrip.(t_Kyrs))) Kyrs", titlesize=50, height=0.0)
                ax0.ylabelvisible = false; ax0.xlabelvisible = false; ax0.xgridvisible = false; ax0.ygridvisible = false;
                ax0.xticksvisible = false; ax0.yticksvisible = false; ax0.yminorticksvisible = false; ax0.xminorticksvisible = false;
                ax0.xgridcolor=:white; ax0.ygridcolor=:white; ax0.ytickcolor=:white; ax0.xtickcolor=:white; ax0.yticklabelcolor=:white;
                ax0.xticklabelcolor=:white; ax0.yticklabelsize=0; ax0.xticklabelsize=0; ax0.xlabelcolor=:white; ax0.ylabelcolor=:white;


                ax1 = Axis(fig[2,1][1,1], aspect = ar, title = L"T [\mathrm{C}]", titlesize=40, yticklabelsize=25, xticklabelsize=25, xlabelsize=25)
                # ax2 = Axis(fig[2,2][1,1], aspect = ar, title = L"\log_{10}(\eta [\mathrm{Pas}])",titlesize=40, yticklabelsize=25, xticklabelsize=25, xlabelsize=25)
                ax2 = Axis(fig[2,2][1,1], aspect = ar, title = L"\log_{10}(\eta_{vep} [\mathrm{Pas}])",titlesize=40, yticklabelsize=25, xticklabelsize=25, xlabelsize=25)
                ax3 = Axis(fig[3,1][1,1], aspect = ar, title = L"Vy [\mathrm{cm/yr}]", titlesize=40, yticklabelsize=25, xticklabelsize=25, xlabelsize=25)
                # ax4 = Axis(fig[3,2][1,1], aspect = ar, title = L"\rho [\mathrm{kgm}^{-3}]", xlabel="Width [km]", titlesize=40, yticklabelsize=25, xticklabelsize=25, xlabelsize=25)
                # ax4 = Axis(fig[3,2][1,1], aspect = ar, title = L"\varepsilon_{\textrm{xy}}[\mathrm{s}^{-1}]", xlabel="Width [km]", titlesize=40, yticklabelsize=25, xticklabelsize=25, xlabelsize=25)
                ax4 = Axis(fig[3,2][1,1], aspect = ar, title = L"\log_{10}(\varepsilon_{\textrm{II}}[\mathrm{s}^{-1}])", xlabel="Width [km]", titlesize=40, yticklabelsize=25, xticklabelsize=25, xlabelsize=25)
                # ax5 = Axis(fig[4,1][1,1], aspect = ar, title = L"\phi", xlabel="Width [km]", titlesize=40, yticklabelsize=25, xticklabelsize=25, xlabelsize=25)
                ax5 = Axis(fig[4,1][1,1], aspect = ar, title = L"P", xlabel="Width [km]", titlesize=40, yticklabelsize=25, xticklabelsize=25, xlabelsize=25)
                ax6 = Axis(fig[4,2][1,1], aspect = ar, title = L"\tau_{\textrm{II}} [MPa]", xlabel="Width [km]", titlesize=40, yticklabelsize=25, xticklabelsize=25, xlabelsize=25)
                # ax6 = Axis(fig[4,2][1,1], aspect = ar, title = L"\tau_{\textrm{xx}} [MPa]", xlabel="Width [km]", titlesize=40, yticklabelsize=25, xticklabelsize=25, xlabelsize=25)
                
                linkyaxes!(ax1,ax2)
                hideydecorations!(ax2, grid = false)

                linkyaxes!(ax3,ax4)
                hideydecorations!(ax4, grid = false)
                # linkyaxes!(ax5,ax6)
                # hideydecorations!(ax6, grid = false)
                
                hidexdecorations!(ax1, grid = false); hidexdecorations!(ax3, grid = false); hidexdecorations!(ax2, grid = false); hidexdecorations!(ax4, grid = false)
                

                # p1  = heatmap!(ax1, x_v, y_v, T_d, colormap=:batlow)
                p1  = heatmap!(ax1, x_c, y_c, T_d, colormap=:batlow)
                      contour!(ax1, x_c, y_c, T_d,; color=:white, levels=600:100:900)
                # p2  = heatmap!(ax2, x_c, y_c, log10.(η_d), Colormap=:roma, ) 
                p2  = heatmap!(ax2, x_c, y_c, log10.(η_vep_d), Colormap=:roma, ) 
                # p2 = heatmap!(ax2, x_v, y_v, phases_new, colormap=:jet)
                # h1 = heatmap!(ax1, x_c, y_c, phase_c, colormap=:jet)     
                # p2  = heatmap!(ax2, x_c, y_c, log10.(η_vep_d), Colormap=:roma, )      
                p3  = heatmap!(ax3, x_v, y_v, Vy_d, colormap=:vik) 
                # p3  = heatmap!(ax3, x_v, y_v, Velo_d, colormap=:vik) 
                # p3  = heatmap!(ax3, x_v, y_v, sqrt.(Vy_v.^2 .+ Vx_v.^2), colormap=:vik)
                # p3  = heatmap!(ax3, x_c, y_c, ∇V_d, colormap=:vik)
                # p4  = heatmap!(ax4, x_c, y_c, Vx_d, colormap=:vik)
                # p4  = heatmap!(ax4, x_c, y_c, ρ_d, colormap=:jet, xlims=(20.0,55.0), ylims=(-20.0, maximum(y_v)))
                p4  = heatmap!(ax4, x_v, y_v, log10.(εII_d), colormap=:jet)#, xlims=(20.0,55.0), ylims=(-20.0, maximum(y_v)))
                # p4  = heatmap!(ax4, x_v, y_v, log10.(εxy_d), colormap=:jet)#, xlims=(20.0,55.0), ylims=(-20.0, maximum(y_v)))
                    # arrows!(ax4, Xp_d[:], Yp_d[:], Vxp[:]*Vscale, Vyp[:]*Vscale, arrowsize = 10, lengthscale=30,arrowcolor=:white ,linecolor=:white)
                # p5  = heatmap!(ax5, x_c, y_c, ϕ_d, colormap=:lajolla)
                p5  = heatmap!(ax5, x_c, y_c, P_d, colormap=:lajolla,xlims=(20.0,55.0), ylims=(-20.0, maximum(y_v)))
                p6  = heatmap!(ax6, x_v, y_v, τII_d, colormap=:batlow)
                # p6  = heatmap!(ax6, x_c, y_c, τxy_d, colormap=:romaO, label="\tau_{\textrm{xy}}")
                # p6  = heatmap!(ax6, x_c, y_c, τxx_d, colormap=:romaO, label="\tau_{\textrm{xy}}")

                Colorbar(fig[2,1][1,2], p1, height=Relative(0.7),ticklabelsize=25, ticksize=15)
                Colorbar(fig[2,2][1,2], p2, height=Relative(0.7),ticklabelsize=25, ticksize=15)
                Colorbar(fig[3,1][1,2], p3, height=Relative(0.7),ticklabelsize=25, ticksize=15)
                Colorbar(fig[3,2][1,2], p4,height=Relative(0.7),ticklabelsize=25, ticksize=15)
                Colorbar(fig[4,1][1,2],p5, height=Relative(0.7),ticklabelsize=25, ticksize=15)
                Colorbar(fig[4,2][1,2],p6, height=Relative(0.7),ticklabelsize=25, ticksize=15)
                #   limits!(ax1, 20.0, 55.0, minimum(y_v), maximum(y_v))
                #   limits!(ax2, 20.0, 55.0, minimum(y_v), maximum(y_v))
                #   limits!(ax3, 20.0, 55.0, minimum(y_v), maximum(y_v))
                #   limits!(ax4, 20.0, 55.0, minimum(y_c), maximum(y_c))
                #   limits!(ax5, 20.0, 55.0, minimum(y_c), maximum(y_c))
                #   limits!(ax6, 20.0, 55.0, minimum(y_c), maximum(y_c))
                
                rowgap!(fig.layout, 1);
                colgap!(fig.layout, 1);
                colgap!(fig.layout, 1);
                
                colgap!(fig.layout, 1);            
                

                fig;
                figsave = joinpath(figdir, @sprintf("%06d.png", it))
                save(figsave, fig)
            end 


            #test plotting
            # Make particles plottable
            p        = particles.coords
            pp = [argmax(p) for p in phase_ratios.center] #if you want to plot it in a heatmap rather than scatter
            ppx, ppy = p
            pxv      = ustrip.(dimensionalize(ppx.data[:],km,CharDim))
            pyv      = ustrip.(dimensionalize(ppy.data[:],km,CharDim))
            clr      = pPhases.data[:]
            idxv     = particles.index.data[:];

            x_v = ustrip.(dimensionalize(xvi[1], km, CharDim))  #not sure about this with MPI and the size (intuition says should be fine)
            y_v = ustrip.(dimensionalize(xvi[2], km, CharDim))
            x_c = ustrip.(dimensionalize(xci[1], km, CharDim))
            y_c = ustrip.(dimensionalize(xci[2], km, CharDim))

            # Make Makie figure
            fig = Figure(resolution = (900, 900), title = "t = $t")
            ax1 = Axis(fig[1,1], aspect = lx/lz, title = "T [K]  (t=$(t/(1e6 * 3600 * 24 *365.25)) Myrs)")
            ax2 = Axis(fig[2,1], aspect = lx/lz, title = "Vy [m/s]")
            ax3 = Axis(fig[1,3], aspect = lx/lz, title = "log10(εII)")
            ax4 = Axis(fig[2,3], aspect = lx/lz, title = "log10(η)")
            # Plot temperature
            h1  = heatmap!(ax1, x_v, y_v, Array(thermal.T[2:end-1,:]) , colormap=:batlow)
            # Plot particles phase
            h2  = scatter!(ax2, Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]))
                
            #  # Find the y-coordinates of the topography
            #     topo_y = findall(x -> x == 1, Phase)

            #  #Find the y-coordinates of the air
            #     air_y = findall(x -> x == 3, Phase)
            #     # Add a line over the scatter plot
            #     line_x = [minimum(pxv[idxv]), maximum(pxv[idxv])]
            #     line_y = [y_v[topo_y[1]], y_v[topo_y[end]]] # y-coordinates of the line
            #     lines!(ax2, line_x, line_y, color=:black, linewidth=2)
            # Plot 2nd invariant of strain rate
            # h3  = heatmap!(ax3, x_c, y_c, Array(log10.(stokes.ε.II)) , colormap=:batlow) 
            # Plot effective viscosity
            # h4  = heatmap!(ax4, x_c, y_c, Array(log10.(η_vep)) , colormap=:batlow)
            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            # hidexdecorations!(ax3)
            Colorbar(fig[1,2], h1)
            Colorbar(fig[2,2], h2)
            # Colorbar(fig[1,4], h3)
            # Colorbar(fig[2,4], h4)
            linkaxes!(ax1, ax2)#, ax3, ax4)
            save(joinpath(figdir, "$(it).png"), fig)
            fig
              # Plot initial T and η profiles
            # if (maximum(Array(stokes.P.-P_init))) <= nondimensionalize(24MPa, CharDim) 
            #     Y =  [y for x in xci[1][2:end-1], y in xci[2][2:end-1]][:]
            #     Y = ustrip.(dimensionalize(Y, km, CharDim))
            #     fig0 = let
            #         fig = Figure(resolution = (1200, 900))
            #         # ax1 = Axis(fig[1,1], aspect = 2/3, title = "T")
            #         ax1 = Axis(fig[1,1], aspect = 2/3, title = "Pressure")
            #         # ax2 = Axis(fig[1,2], aspect = 2/3, title = "log10(η)")
            #         ax2 = Axis(fig[1,2], aspect = 2/3, title = "Density")
            #         lines!(ax1, Array(P_d[:,:][:]), Y)
            #         lines!(ax2, Array(ρ_d[:,:][:]), Y)
            #         ylims!(ax1, minimum(y_c), 0)
            #         ylims!(ax2, minimum(y_c), 0)
            #         hideydecorations!(ax2)
            #         save( joinpath(figdir, "pre_eruptive_profile.png"), fig)
            #         fig
            #     end
            # end

            # if (maximum(Array(stokes.P.-P_init)) >= nondimensionalize(25MPa, CharDim))
            #     Y =  [y for x in xci[1][2:end-1], y in xci[2][2:end-1]][:]
            #     Y = ustrip.(dimensionalize(Y, km, CharDim))
            #     fig0 = let
            #         fig = Figure(resolution = (1200, 900))
            #         # ax1 = Axis(fig[1,1], aspect = 2/3, title = "T")
            #         ax1 = Axis(fig[1,1], aspect = 2/3, title = "Pressure")
            #         # ax2 = Axis(fig[1,2], aspect = 2/3, title = "log10(η)")
            #         ax2 = Axis(fig[1,2], aspect = 2/3, title = "Density")
            #         lines!(ax1, Array(P_d[:,:][:]), Y)
            #         lines!(ax2, Array(ρ_d[:,:][:]), Y)
            #         ylims!(ax1, minimum(y_c), 0)
            #         ylims!(ax2, minimum(y_c), 0)
            #         hideydecorations!(ax2)
            #         save( joinpath(figdir, "eruptive_profile.png"), fig)
            #         fig
            #     end
            # end
            
                    
                    
        #         #   if center[end] < ustrip(nondimensionalize(-25km, CharDim))
        #         #     Angle_rand = [rand(80.0:0.1:100.0)] # Orientation: near-vertical @ depth             
        #         #   else
        #         #     Angle_rand = [rand(-10.0:0.1:10.0)] # Orientation: near-vertical @ shallower depth     
        #         #   end
        #         dike      =   Dike(Angle=Angle_rand, W=W_in, H=H_in, Type=Dike_Type, T=T_in, Center=cen, Phase = 2); 
        #         @copy Tnew_cpu Array(@views thermal.T[2:end-1,:])
        #         ## MTK injectDike
        #         Tracers, Tnew_cpu, Vol, dike_poly, Velo  =   MagmaThermoKinematics.InjectDike(Tracers, Tnew_cpu, xvi, dike, nTr_dike);   # Add dike, move hostrocks

        #         #   if center[end] < ustrip(nondimensionalize(-25km, CharDim))
        #         #     Angle_rand = [rand(80.0:0.1:100.0)] # Orientation: near-vertical @ depth             
        #         #   else
        #         #     Angle_rand = [rand(-10.0:0.1:10.0)] # Orientation: near-vertical @ shallower depth     
        #         #   end
        #         dike      =   Dike(Angle=Angle_rand, W=W_in, H=H_in, Type=Dike_Type, T=T_in, Center=cen, Phase = 2); 
        #         @copy Tnew_cpu Array(@views thermal.T[2:end-1,:])
        #         ## MTK injectDike
        #         Tracers, Tnew_cpu, Vol, dike_poly, Velo  =   MagmaThermoKinematics.InjectDike(Tracers, Tnew_cpu, xvi, dike, nTr_dike);   # Add dike, move hostrocks



            # vtkfile = vtk_grid("$loadpath"*"_$(Int32(it+1e4))", xvi[1],xvi[2]) # 2-D VTK file
            # vtkfile["Temperature"] = Array(T_d); vtkfile["MeltFraction"] = Array(ϕ); 
            # vtkfile["Viscosity"] = Array(η_d); vtkfile["Vy"] = Array(Vy_d); vtkfile["Density"] = Array(ρ_d);vtkfile["Density"] = Array(ρ_d);               # Store fields in file
            # outfiles = vtk_save(vtkfile); #=pvd[time/kyr] = vtkfile=# pvd[ustrip.(t_Myrs)] = vtkfile  
        end



        #     end
        #     if ustrip(dimensionalize(t,yr,CharDim)) > (ustrip(InjectionInterval)*interval)
        #     interval += 1.0 
        #     end
        # end
       
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

    # # to be able to save the history of the simulation - update arrays to be saved accordingly
    # jldsave("history_$figname($it).jld2"; nx = nx, ny = ny, igg = igg, stokes= Array(stokes), thermal=Array(thermal), 
    #         ρg=Array(ρg), η=Array(η), phase_c=Array(phase_c), phase_v=Array(phase_v), ϕ=Array(ϕ), Tracers=Array(Tracers), Tnew_cpu=Array(Tnew_cpu))

    
    # finalize_global_grid(; finalize_MPI=true)
    finalize_global_grid()

    # print_timer() 
    return thermal, stokes
end

# figdir = "figs2D"
# ni, xci, li, di, figdir ,thermal = DikeInjection_2D();
function run()
    figname = "test_PIC"
    ar       = 1 # aspect ratio
    n        = 128
    nx       = n*ar - 2
    ny       = n - 2
    nz       = n-2
    igg      = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
        IGG(init_global_grid(nx, ny, 0; init_MPI= true)...)
        # IGG(init_global_grid(nx, ny, nz; init_MPI= true)...)
    else
        igg
    end
    DikeInjection_2D(igg; figname=figname,nx=nx, ny=ny);

    return figname, nx, ny, igg
    
end

@time run()
