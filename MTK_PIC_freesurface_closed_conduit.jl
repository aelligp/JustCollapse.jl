using CUDA
# CUDA.allowscalar(false) # for safety
CUDA.allowscalar(true) # for safety
using JustRelax, JustRelax.DataIO, JustPIC
import JustRelax.@cell

## NOTE: need to run one of the lines below if one wishes to switch from one backend to another
# set_backend("Threads_Float64_2D")
# set_backend("CUDA_Float64_2D")

const USE_GPU = true;
const GPU_ID = 1;

model = if USE_GPU
    # select if CUDA or AMDGPU
    PS_Setup(:CUDA, Float64, 2)            # initialize parallel stencil in 2D
else
    PS_Setup(:Threads, Float64, 2)            # initialize parallel stencil in 2D
end
environment!(model)

using Printf, Statistics, LinearAlgebra, GeoParams, CairoMakie, SpecialFunctions, CellArrays
using ParallelStencil.FiniteDifferences2D   #specify Dimension
using ImplicitGlobalGrid
using MPI: MPI
using GeophysicalModelGenerator, StaticArrays
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
LonLat = load("./Data/ExportToba_2.jld2", "TobaTopography_LATLON");

proj = ProjectionPoint(; Lat=2.19, Lon=98.91);
Topo = Convert2CartData(LonLat, proj);

println("Done loading Model... starting Dike Injection 2D routine")
#------------------------------------------------------------------------------------------

function init_phases!(phases, particles, phases_topo, xc, yc, a, b, r, xc_anomaly, yc_anomaly, r_anomaly,xc_conduit, yc_conduit, r_conduit)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(
        phases, px, py, index, phases_topo, xc, yc, a, b, r, xc_anomaly, yc_anomaly, r_anomaly,xc_conduit, yc_conduit, r_conduit
    )
        @inbounds for ip in JustRelax.JustRelax.cellaxes(phases)
            # quick escape
            JustRelax.@cell(index[ip, i, j]) == 0 && continue

            x = JustRelax.@cell px[ip, i, j]
            y = -(JustRelax.@cell py[ip, i, j])
            @cell phases[ip, i, j] = 1.0 # crust

            if ((x - xc_conduit)^2 ≤ r_conduit^2) && (0.25*(y - yc_conduit)^2 ≤ r_conduit^2)
                JustRelax.@cell phases[ip, i, j] = 1.0
            end

            # # chamber - elliptical
            if (((x - xc)^2 / ((a)^2)) + ((y + yc)^2 / ((b)^2)) ≤ r^2)
                JustRelax.@cell phases[ip, i, j] = 2.0
            end

            # thermal anomaly - circular
            if ((x - xc_anomaly)^2 + (y + yc_anomaly)^2 ≤ r_anomaly^2)
                JustRelax.@cell phases[ip, i, j] = 3.0
            end

            if y < 0.0
                @cell phases[ip, i, j] = 4.0
            end

        end
        return nothing
    end

    @parallel (JustRelax.@idx ni) init_phases!(
        phases, particles.coords..., particles.index, phases_topo, xc, yc, a, b, r, xc_anomaly, yc_anomaly, r_anomaly, xc_conduit, yc_conduit, r_conduit
    )
end

function update_depth!(depth_corrected, phases, topo_interp, x, y)
    nx, ny = length(x), length(y)

    for i in 1:nx, j in 1:ny

        # vertex coordinates
        xv, yv = x[i], y[j]
        # topography at vertex
        y_topo = topo_interp(xv)
        # depth
        depth = yv

        depth_corrected[i, j] = abs(depth - y_topo)
        if depth > y_topo
            phases[i, j] = 4.0
            # phases[i,j] = Int64(3)
        else
            # phases[i,j] = Int64(1)
            phases[i, j] = 1.0
        end
    end
    return nothing
end

@parallel_indices (i, j) function init_T!(temperature, phases, depth_corrected, geotherm,tempoffset)
    if phases[i, j] == 4.0
        temperature[i + 1, j] = tempoffset
    elseif phases[i, j] == 1.0
        temperature[i + 1, j] = tempoffset + geotherm * @all(depth_corrected)
    end
    return nothing
end

@parallel_indices (i, j) function init_T_particles!(temperature, phases, py, index, geotherm, tempoffset)

    @inbounds for ip in 1:40 #JustRelax.cellaxes(phases)
        # quick escape
        JustRelax.@cell(index[ip, i, j]) == 0 && continue

        # if @cell(phases[ip, i, j]) == 3.0
        #     @cell temperature[ip, i, j] = tempoffset
        # else
            @cell temperature[ip, i, j] = tempoffset + geotherm * abs(@cell(py[ip, i, j]))
        # end
    end
    return nothing
end

@parallel_indices (i, j) function init_P!(pressure, ρg, phases, depth_corrected)
    if phases[i, j] == 4.0
        @all(pressure) = 0.0
    else
        @all(pressure) = 0.0 + @all(ρg) * @all(depth_corrected)
    end
    return nothing
end

function dirichlet_velocities_pureshear!(Vx, Vy, v_extension, xvi)
    lx = abs(reduce(-, extrema(xvi[1])))
    ly = abs(reduce(-, extrema(xvi[2])))
    xv, yv = xvi
    # v_extension /= lx / 2

    @parallel_indices (i, j) function pure_shear_x!(Vx)
        xi = xv[i]
        Vx[i, j + 1] = v_extension * (xi - lx * 0.5) / (lx/2)/2
        # Vx[i, j + 1] = v_extension * (xi - lx * 0.5) / 2
        return nothing
    end

    @parallel_indices (i, j) function pure_shear_y!(Vy)
        yi = abs(yv[j])
        Vy[i + 1, j] = v_extension * yi /ly
        return nothing
    end

    nx, ny = size(Vx)
    @parallel (1:nx, 1:(ny - 2)) pure_shear_x!(Vx)
    nx, ny = size(Vy)
    @parallel (1:(nx - 2), 1:ny) pure_shear_y!(Vy)

    return nothing
end

function circular_anomaly!(T, anomaly, xc, yc, r, xvi)
    @parallel_indices (i, j) function _circular_anomaly!(T, anomaly, xc, yc, r, x, y)
        @inbounds if (((x[i] - xc)^2 + (y[j] - yc)^2) ≤ r^2)
            T[i + 1, j] = anomaly
        end
        return nothing
    end

    ni = length.(xvi)
    @parallel (@idx ni) _circular_anomaly!(T, anomaly, xc, yc, r, xvi...)
    return nothing
end

function elliptical_anomaly!(T, anomaly, xc, yc, a, b, r, xvi)
    @parallel_indices (i, j) function _elliptical_anomaly!(
        T, anomaly, xc, yc, a, b, r, x, y
    )
        @inbounds if (((x[i] - xc)^2 / a^2) + ((y[j] + yc)^2 / b^2) ≤ r^2)
            T[i + 1, j ] = anomaly
        end
        return nothing
    end

    ni = length.(xvi)
    @parallel (@idx ni) _elliptical_anomaly!(T, anomaly, xc, yc, a, b, r, xvi...)
    return nothing
end

function elliptical_anomaly_gradient!(T, offset, xc, yc, a, b, r, xvi)

    @parallel_indices (i, j) function _elliptical_anomaly_gradient!(
        T, offset, xc, yc, a, b, r, x, y
    )
        @inbounds if (((x[i] - xc)^2 / ((a)^2)) + ((y[j] - yc)^2 / ((b)^2)) ≤ r^2)
            T[i+1, j] += offset
            # T[i + 1, j] = rand(0.55:0.001:offset)
        end
        return nothing
    end
    nx, ny = size(T)

    @parallel (1:nx-2, 1:ny) _elliptical_anomaly_gradient!(T, offset, xc, yc, a, b, r, xvi...)
end


@parallel_indices (i, j) function elliptical_anomaly_gradient_particles!(temperature, phases, px, py, index, offset, xc, yc, a, b, r)

    @inbounds for ip in JustRelax.cellaxes(phases)
        # quick escape
        JustRelax.@cell(index[ip, i, j]) == 0 && continue

        @inbounds if (((@cell(px[ip, i, j]) - xc)^2 / a^2) + ((@cell(py[ip, i, j]) + yc)^2 / b^2) ≤ r^2)
            @cell temperature[ip, i, j] = @cell(temperature[ip, i, j]) + offset
        end
    end
    return nothing
end

@parallel_indices (i, j) function conduit_gradient_particles!(temperature, phases, px, py, index, offset, xc_conduit, yc_conduit, r_conduit)

    @inbounds for ip in JustRelax.cellaxes(phases)
        # quick escape
        JustRelax.@cell(index[ip, i, j]) == 0 && continue

        @inbounds if (((@cell(px[ip, i, j]) - xc_conduit)^2 ≤ r_conduit^2) && (0.25*(@cell(py[ip, i, j]) - yc_conduit)^2) ≤ r_conduit^2)
            @cell temperature[ip, i, j] = @cell(temperature[ip, i, j]) + offset

        end
    end
    return nothing
end

function conduit_gradient!(T, offset, xc_conduit, yc_conduit, r_conduit, xvi)

    @parallel_indices (i, j) function _conduit_gradient!(
        T, offset, xc_conduit, yc_conduit, r_conduit, x, y
    )
        @inbounds if ((x[i] - xc_conduit)^2 ≤ r_conduit^2) && (0.325*(y[j] - yc_conduit)^2 ≤ r_conduit^2)
            T[i+1, j] += offset
        end
        return nothing
    end
    nx, ny = size(T)

    @parallel (1:nx-2, 1:ny) _conduit_gradient!(T, offset, xc_conduit, yc_conduit, r_conduit, xvi...)
end

function conduit_gradient_TBuffer!(T, offset, xc_conduit, yc_conduit, r_conduit, xvi)

    @parallel_indices (i, j) function _conduit_gradient_TBuffer!(
        T, offset, xc_conduit, yc_conduit, r_conduit, x, y
    )
        @inbounds if ((x[i] - xc_conduit)^2 ≤ r_conduit^2) && (0.325*(y[j] - yc_conduit)^2 ≤ r_conduit^2)
            T[i, j] += offset
            # T[i + 1, j] = rand(0.55:0.001:offset)
        end
        return nothing
    end
    nx, ny = size(T)

    @parallel (1:nx, 1:ny) _conduit_gradient_TBuffer!(T, offset, xc_conduit, yc_conduit, r_conduit, xvi...)
end



function circular_perturbation!(T, δT, xc_anomaly, yc_anomaly, r_anomaly, xvi)

    @parallel_indices (i, j) function _circular_perturbation!(T, δT, xc_anomaly, yc_anomaly, r_anomaly, x, y)
    @inbounds if  ((x[i] - xc_anomaly)^2 + (y[j] + yc_anomaly)^2 ≤ r_anomaly^2)
        T[i+1, j] *= δT / 100 + 1
    end
    return nothing
end

nx, ny = size(T)

@parallel (1:nx-2, 1:ny) _circular_perturbation!(T, δT, xc_anomaly, yc_anomaly, r_anomaly, xvi...)
end

@parallel_indices (i, j) function circular_perturbation_particles!(temperature, phases, px, py, index, δT, xc, yc, r)

@inbounds for ip in 1:40 #JustRelax.cellaxes(phases)
    # quick escape
    JustRelax.@cell(index[ip, i, j]) == 0 && continue

    @inbounds if (((@cell(px[ip, i, j]) - xc)^2) + ((@cell(py[ip, i, j]) - yc)^2 ) ≤ r^2)
        @cell temperature[ip, i, j] = @cell(temperature[ip, i, j]) * δT / 100 + 1
        #     # T[i + 1, j] = rand(0.55:0.001:offset)
    end
end
return nothing
end


@parallel_indices (i, j) function compute_melt_fraction!(ϕ, rheology, args)
ϕ[i, j] = compute_meltfraction(rheology, ntuple_idx(args, i, j))
return nothing
end

#Multiplies parameter with the fraction of a phase
@generated function compute_param_times_frac(
    fn::F,
    PhaseRatios::Union{NTuple{N,T},SVector{N,T}},
    MatParam::NTuple{N,AbstractMaterialParamsStruct},
    argsi,
    ) where {F,N,T}
    # # Unrolled dot product
    quote
        val = zero($T)
        Base.Cartesian.@nexprs $N i ->
        val += @inbounds PhaseRatios[i] * fn(MatParam[i], argsi)
        return val
    end
end

function compute_meltfraction_ratio(args::Vararg{Any,N}) where {N}
    return compute_param_times_frac(compute_meltfraction, args...)
end

@parallel_indices (I...) function compute_melt_fraction!(ϕ, phase_ratios, rheology, args)
args_ijk = ntuple_idx(args, I...)
ϕ[I...] = compute_melt_frac(rheology, args_ijk, phase_ratios[I...])
return nothing
end

@inline function compute_melt_frac(rheology, args, phase_ratios)
    return compute_meltfraction_ratio(phase_ratios, rheology, args)
end

function copy_arrays_GPU2CPU!(
    T_CPU::AbstractArray, ϕ_CPU::AbstractArray, T_GPU::AbstractArray, ϕ_GPU::AbstractArray
    )
    T_CPU .= Array(T_GPU)
    ϕ_CPU .= Array(ϕ_GPU)

    return nothing
end

function phase_change!(phases, particles)
    ni = size(phases)
    @parallel_indices (I...) function _phase_change!(phases, px, py, index)

        @inbounds for ip in JustRelax.cellaxes(phases)
            #quick escape
            JustRelax.@cell(index[ip, I...]) == 0 && continue

            x = JustRelax.@cell px[ip,I...]
            y = (JustRelax.@cell py[ip,I...])
            phase_ij = @cell phases[ip, I...]
            if y > 0.0 && (phase_ij  == 2.0 || phase_ij  == 3.0)
                @cell phases[ip, I...] = 4.0
            end
        end
        return nothing
    end

    @parallel (JustRelax.@idx ni) _phase_change!( phases, particles.coords..., particles.index)
end

function open_conduit!(phases, particles, xc_conduit, yc_conduit, r_conduit)
    ni = size(phases)
    @parallel_indices (I...) function _open_conduit!(phases, px, py, index, xc_conduit, yc_conduit, r_conduit)

        @inbounds for ip in JustRelax.cellaxes(phases)
            #quick escape
            JustRelax.@cell(index[ip, I...]) == 0 && continue

            x = JustRelax.@cell px[ip,I...]
            y = -(JustRelax.@cell py[ip,I...])

            if ((x - xc_conduit)^2 ≤ r_conduit^2) && (0.25*(y - yc_conduit)^2 ≤ r_conduit^2)
                JustRelax.@cell phases[ip, I...] = 2.0
            end

        end
        return nothing
    end

    @parallel (JustRelax.@idx ni) _open_conduit!(phases, particles.coords..., particles.index, xc_conduit, yc_conduit, r_conduit)
end

solution(ε, t, G, η) = 2 * ε * η * (1 - exp(-G * t / η))

function pureshear!(stokes, εbg, xvi)
    stokes.V.Vx .= PTArray([ x*εbg for x in xvi[1], _ in 1:ny+2])
    stokes.V.Vy .= PTArray([-y*εbg for _ in 1:nx+2, y in xvi[2]])
    return nothing
end


function DikeInjection_2D(igg; figname=figname, nx=nx, ny=ny)

    #-----------------------------------------------------
    # USER INPUTS
    #-----------------------------------------------------
    Topography= false; #specify if you want topography plotted in the figures
    Freesurface = true #specify if you want to use freesurface
        sticky_air = 5km #specify the thickness of the sticky air layer
    Inject_Dike = false #specify if you want to inject a dike
    dike_width = 2.5km  #specify the width of the dike
    dike_height = 1.0km #specify the height of the dike
    dike_temp = 1000C   #specify the temperature of the dike
    InjectionInterval = 25yr        # number of timesteps between injections (if Inject_Dike=true). Increase this at higher resolution.
    #now injecting every 2 it

    toy = true #specify if you want to use the toy model or the Toba model
    thermal_perturbation = :elliptical_anomaly #specify if you want a thermal perturbation in the center of the domain or random
    sphere = 15km                               #specify the radius of the circular anomaly
    ellipse = 35km
    temp_anomaly = 1000C

    shear = true #specify if you want to use pure shear boundary conditions
    # v_extension = 1e-10m/s
    regime = ViscoElastic                  #Rheology of the Stokes Solver: ViscoElastic, ViscoElastoPlastic

    Nx, Ny, Nz = 100, 100, 100      # 3D grid size (does not yet matter)
    arrow_steps_x_dir = 5           # number of arrows in the quiver plot
    arrow_steps_y_dir = 4           # number of arrows in the quiver plot

    nt = 500                        # number of timesteps

    η_uppercrust = 1e21             #viscosity of the upper crust
    η_magma = 1e16                  #viscosity of the magma
    η_air = 1e16                    #viscosity of the air

    #-----------------------------------------------------

    # IO ------------------------------------------------
    figdir = "./fig2D/$figname/"
    # gifname = "$figname";
    pvd = paraview_collection("./Paraview/$figname")
    !isdir(figdir) && mkpath(figdir)
    # anim = Animation(figdir, String[]);
    # Preparation of VTK/Paraview output
    # if isdir("./Paraview/$figname/")==false mkdir("./Paraview/$figname/") end; loadpath = "./Paraview/$figname/";

    # Standard MTK Toba routine--------------------------------------------------------------

    # nondimensionalize with CharDim
    CharDim = GEO_units(; length=40km, viscosity=1e20Pa * s)
    ni = (nx, ny)   #grid spacing for calculation (without ghost cells)

    Grid = CreateCartGrid(;
        size=(Nx, Ny, Nz),
        x=((Topo.x.val[1, 1, 1])km, (Topo.x.val[end, 1, 1])km),
        y=((Topo.y.val[1, 1, 1])km, (Topo.y.val[1, end, 1])km),
        z=(-40km, 4km),
    )
    X, Y, Z = XYZGrid(Grid.coord1D...)
    DataTopo = CartData(X, Y, Z, (Depthdata=Z,))

    Lon, Lat, Depth = (Topo.x.val .* km), (Topo.y.val .* km), ((Topo.z.val ./ 1e3) .* km)
    Topo_Cart = CartData(Lon, Lat, Depth, (Depth=Depth,))

    ind = AboveSurface(DataTopo, Topo_Cart)
    Phase = ones(size(X))
    Phase[ind] .= 3

    DataPara = CartData(X, Y, Z, (Phase=Phase,))
    Phase = Int64.(round.(DataPara.fields.Phase))

    # ----------CrossSections for 2D simulation----------------------------------


    Data_Cross = CrossSection(
        DataPara;
        dims=(ni[1] + 1, ni[2] + 1),
        Interpolate=true,
        Start=(ustrip(-40.00km), ustrip(50.00km)),
        End=(ustrip(-06.00km), ustrip(20.00km)),
    )

    x_new = FlattenCrossSection(Data_Cross)
    Data_Cross = AddField(Data_Cross, "FlatCrossSection", x_new)
    Phase = dropdims(Data_Cross.fields.Phase; dims=3) #Dropped the 3rd dim of the Phases to be consistent with new grid of the CrossSection
    Phase = Int64.(round.(Phase))
    Phase[Phase .== 2] .= 3 #set the phase of the air to 3

    Topo_Cross = CrossSection(
        Topo_Cart;
        dims=(ni[1] + 1, ni[2] + 1),
        Interpolate=true,
        Start=(ustrip(-40.00km), ustrip(50.00km)),
        End=(ustrip(-06.00km), ustrip(20.00km)),
    )

    x_new_Topo = FlattenCrossSection(Topo_Cross)
    Topo_Cross = AddField(Topo_Cross, "FlatCrossSection", x_new_Topo)
    Topo_nondim = nondimensionalize(Topo_Cross.fields.Depth, CharDim)

    ## Workaround to display topo on heatmaps for η,ρ,ϕ due to different dimensions (ni, not ni+1)
    Data_Cross_ni = CrossSection(
        DataPara;
        dims=(ni[1], ni[2]),
        Interpolate=true,
        Start=(ustrip(-40.00km), ustrip(50.00km)),
        End=(ustrip(-06.00km), ustrip(20.00km)),
    )

    Phase_ni = dropdims(Data_Cross_ni.fields.Phase; dims=3) #Dropped the 3rd dim of the Phases to be consistent with new grid of the CrossSection
    Phase_ni = Int64.(round.(Phase_ni))
    Phase_ni[Phase_ni .== 2] .= 3 #set the phase of the air to 3
    #Seismo Model
    Model3D_Cross = CrossSection(
        Model3D_cart;
        dims=(ni[1] + 1, ni[2] + 1),
        Interpolate=true,
        Start=(ustrip(-40.00km), ustrip(50.00km)),
        End=(ustrip(-06.00km), ustrip(20.00km)),
    )

    Model3D_new = FlattenCrossSection(Model3D_Cross)
    Model3D_Cross = AddField(Model3D_Cross, "Model3D_Cross", Model3D_new)

    #seismo model with ni dimensions
    Model3D_Cross_ni = CrossSection(
        Model3D_cart;
        dims=(ni[1], ni[2]),
        Interpolate=true,
        Start=(ustrip(-40.00km), ustrip(50.00km)),
        End=(ustrip(-06.00km), ustrip(20.00km)),
    )

    Model3D_new_ni = FlattenCrossSection(Model3D_Cross_ni)
    Model3D_Cross_ni = AddField(Model3D_Cross_ni, "Model3D_Cross", Model3D_new_ni)

    #New 2D Grid
    if Topography == true
        Grid2D = CreateCartGrid(;
            size=(ni[1], ni[2]),
            x=(extrema(Data_Cross.fields.FlatCrossSection) .* km),
            z=(
                (minimum(Data_Cross.z.val) ./ 2.66) .* km, (maximum(Data_Cross.z.val)) .* km
            ),
            CharDim=CharDim,
        ) #create new 2D grid for Injection routine
    elseif Freesurface == true
        # new grid with no topography as the topo now seems to be the problem with strain rate.

        Grid2D = CreateCartGrid(;
            size=(ni[1], ni[2]),
            x=(extrema(Data_Cross.fields.FlatCrossSection) .* km),
            z=((-25.0) .* km, sticky_air),
            CharDim=CharDim,
        ) #create new 2D grid for Injection routine
    else
        # new grid with no topography as the topo now seems to be the problem with strain rate.

        Grid2D = CreateCartGrid(;
            size=(ni[1], ni[2]),
            x=(extrema(Data_Cross.fields.FlatCrossSection) .* km),
            z=((-25.0) .* km, 0.0 .* km),
            CharDim=CharDim,
        ) #create new 2D grid for Injection routine
    end

    Phi_melt_data = dropdims(Model3D_Cross.fields.Phi_melt; dims=3)
    Phi_melt_data_ni = dropdims(Model3D_Cross_ni.fields.Phi_melt; dims=3)

    #Dike location initiation
    ind_melt = findall(Phi_melt_data .> 0.12) # only Meltfraction of 12% used for dike injection
    ind_melt_ni = findall(Phi_melt_data_ni .> 0.12) # only Meltfraction of 12% used for dike injection
    x1, z1 = dropdims(Data_Cross.fields.FlatCrossSection; dims=3),
    dropdims(Data_Cross.z.val; dims=3) # Define x,z coord for injection
    x1, z1 = x1 .* km, z1 .* km # add dimension to FlatCrossSection to non-dim x,z for consistency
    x1, z1 = nondimensionalize(x1, CharDim), nondimensionalize(z1, CharDim)

    #-------rheology parameters--------------------------------------------------------------
    # plasticity setup
    do_DP   = true               # do_DP=false: Von Mises, do_DP=true: Drucker-Prager (friction angle)
    η_reg   = 1.0e14Pas           # regularisation "viscosity" for Drucker-Prager
    Coh     = 10MPa              # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ       = 30.0 * do_DP         # friction angle
    G0      = 25e9Pa        # elastic shear modulus
    G_magma = 10e9Pa        # elastic shear modulus perturbation
    εbg     = 1e-15 / s             # background strain rate
    εbg     = nondimensionalize(εbg, CharDim) # background strain rate

    pl          = DruckerPrager_regularised(; C=Coh, ϕ=ϕ, η_vp=η_reg, Ψ=0)        # plasticity

    el       = SetConstantElasticity(; G=G0, ν=0.5)                            # elastic spring
    el_magma = SetConstantElasticity(; G=G_magma, ν=0.3)                            # elastic spring
    el_air   = SetConstantElasticity(; ν=0.5, Kb=0.101MPa)                            # elastic spring
    disl_upper_crust = DislocationCreep(;
        A=5.07e-18, n=2.3, E=154e3, V=0.0, r=0.0, R=8.3145
    ) #(;E=187kJ/mol,) Kiss et al. 2023
    creep_rock = LinearViscous(; η=η_uppercrust * Pa * s)
    creep_magma = LinearViscous(; η=η_magma * Pa * s)
    creep_air = LinearViscous(; η=η_air * Pa * s)
    cutoff_visc = (
        nondimensionalize(1e14Pa * s, CharDim), nondimensionalize(1e24Pa * s, CharDim)
    )
    β_rock = inv(get_Kb(el))
    β_magma = inv(get_Kb(el_magma))
    Kb = get_Kb(el)

    #-------JustRelax parameters-------------------------------------------------------------
    # Domain setup for JustRelax
    lx, lz = (Grid2D.L[1]), (Grid2D.L[2]) # nondim if CharDim=CharDim
    li = lx, lz
    b_width = (4, 4, 0) #boundary width
    origin = Grid2D.min[1], Grid2D.min[2]
    igg = igg
    di = @. li / ni # grid step in x- and y-direction
    # di = @. li / (nx_g(), ny_g()) # grid step in x- and y-direction
    xci, xvi = lazy_grid(di, li, ni; origin=origin) #non-dim nodes at the center and the vertices of the cell (staggered grid)
    #---------------------------------------------------------------------------------------

    # Set material parameters
    MatParam = (
        #Name="UpperCrust"
        SetMaterialParams(;
            Phase   = 1,
            Density  = PT_Density(ρ0=2700kg/m^3, β=β_rock/Pa),
            HeatCapacity = ConstantHeatCapacity(cp=1050J/kg/K),
            Conductivity = ConstantConductivity(k=3.0Watt/K/m),
            LatentHeat = ConstantLatentHeat(Q_L=350e3J/kg),
            CompositeRheology = CompositeRheology((creep_rock, el, pl, )),
            Melting = MeltingParam_Caricchi(),
            Elasticity = el,
            CharDim  = CharDim,),

        #Name="Magma"
        SetMaterialParams(;
            Phase   = 2,
            Density  = PT_Density(ρ0=2600kg/m^3, β=β_magma/Pa),
            HeatCapacity = ConstantHeatCapacity(cp=1050J/kg/K),
            Conductivity = ConstantConductivity(k=1.5Watt/K/m),
            LatentHeat = ConstantLatentHeat(Q_L=350e3J/kg),
            CompositeRheology = CompositeRheology((creep_magma, el_magma)),
            Melting = MeltingParam_Caricchi(),
            # Elasticity = el_magma,
            CharDim  = CharDim,),

        #Name="Thermal Anomaly"
        SetMaterialParams(;
            Phase   = 3,
            Density  = PT_Density(ρ0=2600kg/m^3, β=β_magma/Pa),
            HeatCapacity = ConstantHeatCapacity(cp=1050J/kg/K),
            Conductivity = ConstantConductivity(k=1.5Watt/K/m),
            LatentHeat = ConstantLatentHeat(Q_L=350e3J/kg),
            CompositeRheology = CompositeRheology((creep_magma,el_magma)),
            Melting = MeltingParam_Caricchi(),
            # Elasticity = el_magma,
            CharDim  = CharDim,),

        #Name="Sticky Air"
        SetMaterialParams(;
            Phase   = 4,
            Density   = ConstantDensity(ρ=100kg/m^3,),
            HeatCapacity = ConstantHeatCapacity(cp=1000J/kg/K),
            Conductivity = ConstantConductivity(k=15Watt/K/m),
            LatentHeat = ConstantLatentHeat(Q_L=0.0J/kg),
            CompositeRheology = CompositeRheology((creep_air,)),
            # Elasticity = ConstantElasticity(; G=Inf*Pa, Kb=Inf*Pa),
            CharDim = CharDim),
            )

    #----------------------------------------------------------------------------------

    #update depth with topography !!!CPU ONLY!!!
    phases_topo_v = zeros(nx + 1, ny + 1)
    phases_topo_c = zeros(nx, ny)
    depth_corrected_v = zeros(nx + 1, ny + 1)
    depth_corrected_c = zeros(nx, ny)

    if Topography == false
        Topo_nondim .= 0.0
    end
    topo_interp = linear_interpolation(xvi[1], Topo_nondim)
    update_depth!(depth_corrected_v, phases_topo_v, topo_interp, xvi...)

    depth_corrected_v = PTArray(depth_corrected_v)
    depth_corrected_c = PTArray(depth_corrected_c)
    phases_topo_v = (PTArray(phases_topo_v))
    phases_topo_c = (PTArray(phases_topo_c))
    @parallel vertex2center!(phases_topo_c, phases_topo_v)
    @parallel vertex2center!(depth_corrected_c, depth_corrected_v)
    @views phases_topo_c = round.(phases_topo_c)
    @views phases_topo_c[phases_topo_c .== 2.0] .= 3.0

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 20, 40, 5 # nxcell = initial number of particles per cell; max_cell = maximum particles accepted in a cell; min_xcell = minimum number of particles in a cell
    particles = init_particles_cellarrays(
        nxcell, max_xcell, min_xcell, xvi..., di..., ni...
    )
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases = init_particle_fields_cellarrays(particles, Val(3))
    particle_args = (pT, pPhases)

    if Topography == true
        xc, yc = 0.66 * lx, 0.4 * -lz  # origin of thermal anomaly
        x_anomaly, y_anomaly = lx * 0.66, -lz * 0.5  # Randomly vary center of dike
    elseif Freesurface == true
        xc, yc = 0.5 * lx, 0.5 * -(lz-nondimensionalize(sticky_air,CharDim))  # origin of thermal anomaly
        x_anomaly, y_anomaly = lx * 0.5, -(lz-nondimensionalize(sticky_air,CharDim)) * 0.6 # Randomly vary center of dike
    else
        xc, yc = 0.5 * lx, 0.5 * -lz # origin of thermal anomaly
        x_anomaly, y_anomaly = lx * 0.5, -lz * 0.6 # Randomly vary center of dike
    end

    radius = nondimensionalize(ellipse, CharDim)         # radius of perturbation
    a = nondimensionalize(15km, CharDim)
    b = nondimensionalize(5km, CharDim)
    r_anomaly = nondimensionalize(1.5km,CharDim)             # radius of perturbation
    xc_conduit, yc_conduit, r_conduit = (lx*0.5),nondimensionalize(4.15km,CharDim),nondimensionalize(2.3km,CharDim)

    init_phases!(pPhases, particles, phases_topo_v, xc, yc, a, b, radius, x_anomaly, y_anomaly, r_anomaly, xc_conduit, yc_conduit, r_conduit)
    phase_ratios = PhaseRatio(ni, length(MatParam))
    @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases)

    # Physical Parameters
    geotherm = GeoUnit(30K / km)
    geotherm_nd = ustrip(Value(nondimensionalize(geotherm, CharDim)))
    ΔT = geotherm_nd * (lz - nondimensionalize(sticky_air,CharDim)) # temperature difference between top and bottom of the domain
    tempoffset = nondimensionalize(0C, CharDim)
    η = MatParam[2].CompositeRheology[1][1].η.val       # viscosity for the Rayleigh number
    Cp0 = MatParam[2].HeatCapacity[1].cp.val              # heat capacity
    ρ0 = MatParam[2].Density[1].ρ0.val                   # reference Density
    k0 = MatParam[2].Conductivity[1]              # Conductivity
    G = MatParam[1].Elasticity[1].G.val                 # Shear Modulus
    κ = nondimensionalize(1.5Watt / K / m, CharDim) / (ρ0 * Cp0)                                   # thermal diffusivity
    g = MatParam[1].Gravity[1].g.val                    # Gravity

    α = MatParam[1].Density[1].α.val                    # thermal expansion coefficient for PT Density
    Ra =   ρ0 * g * α * ΔT * 10^3 / (η * κ)                # Rayleigh number
    dt = dt_diff = 0.5 * min(di...)^2 / κ / 2.01           # diffusive CFL timestep limiter

    v_extension = nondimensionalize(2.0cm / yr, CharDim)   # extension velocity for pure shear boundary conditions
    relaxation_time =
        nondimensionalize(η_magma * Pas, CharDim) / nondimensionalize(G_magma, CharDim) # η_magma/Gi
    # Initialize arrays for PT thermal solver
    k = @fill(nondimensionalize(1.5Watt / K / m, CharDim), ni...)
    ρCp = @fill(ρ0 .* Cp0, ni...)

    # Initialisation
    thermal = ThermalArrays(ni)                                # initialise thermal arrays and boundary conditions
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux=(left=true, right=true, top=false, bot=false),
        periodicity=(left=false, right=false, top=false, bot=false),
    )

    if Topography == true
        depth_corrected_dim = ustrip.(dimensionalize(depth_corrected_v, km, CharDim))
    else
        xyv = PTArray([y for x in xvi[1], y in xvi[2]])
        depth_corrected_dim = ustrip.(dimensionalize(-xyv, km, CharDim))
    end

    @parallel (@idx ni .+ 1) init_T!(
        thermal.T, phases_topo_v, depth_corrected_v, geotherm_nd,tempoffset
    )

    thermal_bcs!(thermal.T, thermal_bc)

    @parallel (JustRelax.@idx size(thermal.Tc)...) temperature2center!(
        thermal.Tc, thermal.T
    )

    Tnew_cpu = Array{Float64}(undef, ni .+ 1...)                  # Temperature for the CPU
    Phi_melt_cpu = Array{Float64}(undef, ni...)                    # Melt fraction for the CPU

    stokes = StokesArrays(ni, ViscoElastic)                         # initialise stokes arrays with the defined regime
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-4, CFL=0.99 / √2.1) #ϵ=1e-4,  CFL=1 / √2.1 CFL=0.27 / √2.1

    args = (; T=thermal.Tc, P=stokes.P, dt=Inf)

    pt_thermal = PTThermalCoeffs(
        MatParam, phase_ratios, args, dt, ni, di, li; ϵ=1e-5, CFL=1e-3 / √2.1
    )
    # Boundary conditions of the flow
    if shear == true
        dirichlet_velocities_pureshear!(@velocity(stokes)..., v_extension, xvi)
        # pureshear_bc!(stokes, xci, xvi, εbg)
    end
    flow_bcs = FlowBoundaryConditions(;
        free_slip=(left=true, right=true, top=true, bot=true)
    )
    flow_bcs!(stokes, flow_bcs)

    η = @ones(ni...)                                     # initialise viscosity
    η_vep = deepcopy(η)                                       # initialise viscosity for the VEP
    G = @fill(MatParam[1].Elasticity[1].G.val, ni...)     # initialise shear modulus
    ϕ = similar(η)                                        # melt fraction center

    # Buoyancy force
    ρg = @zeros(ni...), @zeros(ni...)                      # ρg[1] is the buoyancy force in the x direction, ρg[2] is the buoyancy force in the y direction

    P_init = @zeros(ni...)
    P_dike = @zeros(ni...)
    P_chamber = @zeros(ni...)
    Qmask = @zeros(ni...)
    Qmass = 0.15
    P_Dirichlet = @zeros(ni...)
    SH = @zeros(ni...) #shear heating to be Updated

    # Preparation for Visualisation
    ni_v_viz  = nx_v_viz, ny_v_viz = (ni[1] - 1) * igg.dims[1], (ni[2] - 1) * igg.dims[2]      # size of the visualisation grid on the vertices according to MPI dims
    ni_viz    = nx_viz, ny_viz = (ni[1] - 2) * igg.dims[1], (ni[2] - 2) * igg.dims[2]            # size of the visualisation grid on the vertices according to MPI dims
    Vx_vertex = PTArray(ones(ni .+ 1...))                                                  # initialise velocity for the vertices in x direction
    Vy_vertex = PTArray(ones(ni .+ 1...))                                                  # initialise velocity for the vertices in y direction

    # Arrays for visualisation
    Tc_viz    = Array{Float64}(undef,ni_viz...)                                   # Temp center with ni
    Vx_viz    = Array{Float64}(undef,ni_v_viz...)                                 # Velocity in x direction with ni_viz .-1
    Vy_viz    = Array{Float64}(undef,ni_v_viz...)                                 # Velocity in y direction with ni_viz .-1
    ∇V_viz    = Array{Float64}(undef,ni_viz...)                                   # Velocity in y direction with ni_viz .-1
    P_viz     = Array{Float64}(undef,ni_viz...)                                   # Pressure with ni_viz .-2
    τxy_viz   = Array{Float64}(undef,ni_v_viz...)                                 # Shear stress with ni_viz .-1
    τII_viz   = Array{Float64}(undef,ni_viz...)                                   # 2nd invariant of the stress tensor with ni_viz .-2
    εII_viz   = Array{Float64}(undef,ni_viz...)                                   # 2nd invariant of the strain tensor with ni_viz .-2
    εxy_viz   = Array{Float64}(undef,ni_v_viz...)                                 # Shear strain with ni_viz .-1
    η_viz     = Array{Float64}(undef,ni_viz...)                                   # Viscosity with ni_viz .-2
    η_vep_viz = Array{Float64}(undef,ni_viz...)                                   # Viscosity for the VEP with ni_viz .-2
    ϕ_viz     = Array{Float64}(undef,ni_viz...)                                   # Melt fraction with ni_viz .-2
    ρg_viz    = Array{Float64}(undef,ni_viz...)                                   # Buoyancy force with ni_viz .-2

    # Arguments for functions
    args = (; ϕ=ϕ, T=thermal.Tc, P=stokes.P, dt=dt)
    @copy thermal.Told thermal.T
    @copy Tnew_cpu Array(thermal.T[2:(end - 1), :])

    for _ in 1:2
        @parallel (JustRelax.@idx ni) compute_ρg!(
            ρg[2], phase_ratios.center, MatParam, (T=thermal.Tc, P=stokes.P)
        )
        @parallel (@idx ni) init_P!(stokes.P, ρg[2], phases_topo_c, depth_corrected_c)
    end

    @parallel (@idx ni) compute_viscosity!(
        η, 1.0, phase_ratios.center, @strain(stokes)..., args, MatParam, cutoff_visc
    )
    η_vep = copy(η)

    pressure_top = nondimensionalize(0.0MPa,CharDim)
    if toy == true
        if thermal_perturbation == :random
            δT = 5.0              # thermal perturbation (in %)
            random_perturbation!(
                thermal.T, δT, (lx * 1 / 8, lx * 7 / 8), (-2000e3, -2600e3), xvi
            )
        elseif thermal_perturbation == :circular
            δT = 10.0              # thermal perturbation (in %)
            r  = nondimensionalize(5km, CharDim)         # radius of perturbation
            circular_perturbation!(thermal.T, δT, xc, yc, r, xvi)

        elseif thermal_perturbation == :circular_anomaly
            anomaly = nondimensionalize(temp_anomaly, CharDim) # temperature anomaly
            radius  = nondimensionalize(sphere, CharDim)         # radius of perturbation
            circular_anomaly!(thermal.T, anomaly, xc, yc, radius, xvi)

        elseif thermal_perturbation == :elliptical_anomaly
            anomaly = nondimensionalize(temp_anomaly, CharDim) # temperature anomaly
            radius  = nondimensionalize(ellipse, CharDim)         # radius of perturbation
            offset  = nondimensionalize(150C, CharDim)
            δT      = 20.0              # thermal perturbation (in %)
            elliptical_anomaly_gradient!(
                    thermal.T, offset, xc, yc, a, b, radius, xvi
                    )
            # conduit_gradient!(thermal.T, offset, xc_conduit, -yc_conduit, r_conduit, xvi)
            circular_perturbation!(thermal.T, δT, x_anomaly, -y_anomaly, r_anomaly, xvi)
        end
    end
    # make sure they are the same
    thermal.Told .= thermal.T
    @parallel (JustRelax.@idx size(thermal.Tc)...) temperature2center!(
        thermal.Tc, thermal.T
    )
    @parallel (@idx ni) compute_melt_fraction!(
        ϕ, phase_ratios.center, MatParam, (T=thermal.Tc, P=stokes.P)
    )

    # Time loop
    t, it      = 0.0, 0
    interval   = 1.0
    InjectVol  = 0.0
    evo_t      = Float64[]
    evo_InjVol = Float64[]
    local iters

    T_buffer    = @zeros(ni .+ 1)
    Told_buffer = similar(T_buffer)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end

    grid2particle!(pT, xvi, T_buffer, particles.coords)
    p = particles.coords
    pp = PTArray([argmax(p) for p in phase_ratios.center]) #if you want to plot it in a heatmap rather than scatter
    mask_sticky_air = @zeros(ni.+1...)
    @parallel center2vertex!(mask_sticky_air, pp)
    @copy stokes.P0 stokes.P

    # Plot initial T and η profiles
    let
        Yv = [y for x in xvi[1], y in xvi[2]][:]
        Y = [y for x in xci[1], y in xci[2]][:]
        fig = Figure(; resolution=(1200, 900))
        ax1 = Axis(fig[1, 1]; aspect=2 / 3, title="T")
        ax2 = Axis(fig[1, 2]; aspect=2 / 3, title="Pressure")
        scatter!(
            ax1,
            Array(ustrip.(dimensionalize(thermal.T[2:(end - 1), :][:], C, CharDim))),
            ustrip.(dimensionalize(Yv, km, CharDim)),
        )
        lines!(
            ax2,
            Array(ustrip.(dimensionalize(stokes.P[:], MPa, CharDim))),
            ustrip.(dimensionalize(Y, km, CharDim)),
        )
        hideydecorations!(ax2)
        save(joinpath(figdir, "initial_profile.png"), fig)
        fig
    end

    dt *= 0.2
    while it < 15 #nt

        particle2grid!(T_buffer, pT, xvi, particles.coords)
        @views T_buffer[:, end] .= nondimensionalize(0.0C, CharDim)
        # @views T_buffer[args.sticky_air.==4.0] .= nondimensionalize(0.0C, CharDim)
        # @views T_buffer[:, 1] .= maximum(thermal.T)
        @views thermal.T[2:end-1, :] .= T_buffer
        temperature2center!(thermal)

        args = (; ϕ=ϕ, T=thermal.Tc, P=stokes.P, dt=dt, pressure_top=pressure_top,sticky_air=mask_sticky_air)

        #open the conduit
        # if it == 5
        #     conduit_gradient_TBuffer!(T_buffer, offset, xc_conduit, -yc_conduit, r_conduit, xvi)
        #     open_conduit!(pPhases, particles, xc_conduit, yc_conduit, r_conduit)
        #     grid2particle_flip!(pT, xvi, T_buffer, Told_buffer, particles.coords)
        # end

        @views T_buffer[:, end] .= nondimensionalize(0.0C, CharDim)
        @views T_buffer[args.sticky_air.==4.0] .= nondimensionalize(0.0C, CharDim)
        # @views T_buffer[:, 1] .= maximum(thermal.T)
        @views thermal.T[2:end-1, :] .= T_buffer

        temperature2center!(thermal)
        args = (; ϕ=ϕ, T=thermal.Tc, P=stokes.P, dt=dt, pressure_top=pressure_top,sticky_air=mask_sticky_air)
        @parallel (@idx ni) compute_viscosity!(
            η, 1.0, phase_ratios.center, @strain(stokes)..., args, MatParam, cutoff_visc
        )
        @parallel (@idx ni) compute_ρg!(
            ρg[2], phase_ratios.center, MatParam, (T=thermal.Tc, P=stokes.P)
        )
        @copy stokes.P0 stokes.P
        args = (; ϕ=ϕ, T=thermal.Tc, P=stokes.P, dt=dt, pressure_top=pressure_top, sticky_air=mask_sticky_air)#, S=S, mfac=mfac, η_f=η_f, η_s=η_s)
        # Stokes solver -----------------
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
            dt * 1e-1,
            igg;
            iterMax = 175e3,
            nout = 1e3,
            b_width,
            viscosity_cutoff=cutoff_visc,
        )
        @parallel (JustRelax.@idx ni) tensor_invariant!(stokes.ε.II, @strain(stokes)...)
        @parallel (@idx ni .+ 1) multi_copy!(@tensor(stokes.τ_o), @tensor(stokes.τ))
        @parallel (@idx ni) multi_copy!(
            @tensor_center(stokes.τ_o), @tensor_center(stokes.τ)
        )
        dt = compute_dt(stokes, di, dt_diff, igg) * 0.2
        # if it < 5
        #     dt *= 0.1
        # else
        #     dt *= 1e-2
        # end
        # ------------------------------
        @show dt
        # @show extrema(stokes.V.Vy)
        # @show extrema(stokes.V.Vx)

        # Thermal solver ---------------
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            MatParam,
            args,
            dt,
            di;
            igg=igg,
            phase=phase_ratios,
            iterMax=150e3,
            nout=1e3,
            verbose=true,
        )

        for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
            copyinn_x!(dst, src)
        end

        # Advection --------------------
        # advect particles in space
        advection_RK!(particles, @velocity(stokes), grid_vx, grid_vy, dt, 2 / 3)
        # advect particles in memory
        shuffle_particles!(particles, xvi, particle_args)
        # JustPIC.clean_particles!(particles, xvi, particle_args)
        grid2particle_flip!(pT, xvi, T_buffer, Told_buffer, particles.coords)
        println("Check 1, after grid2particle_flip: ", extrema((pT.data[:])[particles.index.data[:]]))
        # check if we need to inject particles
        inject = check_injection(particles)
        inject && inject_particles_phase!(particles, pPhases, (pT, ), (T_buffer,), xvi)

        #phase change for particles
        phase_change!(pPhases, particles)
        # update phase ratios
        @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases)
        @parallel (@idx ni) compute_melt_fraction!(
            ϕ, phase_ratios.center, MatParam, (T=thermal.Tc, P=stokes.P)
        )

        @show it += 1
        t += dt

        #  # # Plotting -------------------------------------------------------
        if it == 1 || rem(it, 1) == 0
            checkpointing(figdir, stokes, thermal.T, η, t)
            # Arrow plotting routine
            Xc, Yc = [x for x in xvi[1], y in xvi[2]], [y for x in xvi[1], y in xvi[2]]
            st_x = arrow_steps_x_dir                                       # quiver plotting spatial step x direction
            st_y = arrow_steps_y_dir                                       # quiver plotting spatial step y direction
            Xp, Yp = Xc[1:st_x:end, 1:st_y:end], Yc[1:st_x:end, 1:st_y:end]
            Vxp = Array(
                0.5 * (
                    stokes.V.Vx[1:st_x:(end - 1), 1:st_y:end] +
                    stokes.V.Vx[2:st_x:end, 1:st_y:end]
                ),
            )
            Vyp = Array(
                0.5 * (
                    stokes.V.Vy[1:st_x:end, 1:st_y:(end - 1)] +
                    stokes.V.Vy[1:st_x:end, 2:st_y:end]
                ),
            )
            Vscale = 0.5 / maximum(sqrt.(Vxp .^ 2 + Vyp .^ 2)) * di[1] * (st_x - 1)

            JustRelax.velocity2vertex!(Vx_vertex, Vy_vertex, stokes.V.Vx, stokes.V.Vy)
            Xp_d = Array(ustrip.(dimensionalize(Xp, km, CharDim)))
            Yp_d = Array(ustrip.(dimensionalize(Yp, km, CharDim)))

            x_v = ustrip.(dimensionalize(xvi[1][2:(end - 1)], km, CharDim))  #not sure about this with MPI and the size (intuition says should be fine)
            y_v = ustrip.(dimensionalize(xvi[2][2:(end - 1)], km, CharDim))
            x_c = ustrip.(dimensionalize(xci[1][2:(end - 1)], km, CharDim))
            y_c = ustrip.(dimensionalize(xci[2][2:(end - 1)], km, CharDim))

            T_inn = Array(thermal.Tc[2:(end - 1), 2:(end - 1)])
            Vx_inn = Array(Vx_vertex[2:(end - 1), 2:(end - 1)])
            Vy_inn = Array(Vy_vertex[2:(end - 1), 2:(end - 1)])
            ∇V_inn = Array(stokes.∇V[2:(end - 1), 2:(end - 1)])
            P_inn = Array(stokes.P[2:(end - 1), 2:(end - 1)])
            τII_inn = Array(stokes.τ.II[2:(end - 1), 2:(end - 1)])
            τxy_inn = Array(stokes.τ.xy[2:(end - 1), 2:(end - 1)])
            εII_inn = Array(stokes.ε.II[2:(end - 1), 2:(end - 1)])
            εxy_inn = Array(stokes.ε.xy[2:(end - 1), 2:(end - 1)])
            η_inn = Array(η[2:(end - 1), 2:(end - 1)])
            η_vep_inn = Array(η_vep[2:(end - 1), 2:(end - 1)])
            ϕ_inn = Array(ϕ[2:(end - 1), 2:(end - 1)])
            ρg_inn = Array(ρg[2][2:(end - 1), 2:(end - 1)])

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

            T_d = ustrip.(dimensionalize(Array(Tc_viz), C, CharDim))
            η_d = ustrip.(dimensionalize(Array(η_viz), Pas, CharDim))
            η_vep_d = ustrip.(dimensionalize(Array(η_vep_viz), Pas, CharDim))
            Vy_d = ustrip.(dimensionalize(Array(Vy_viz), cm / yr, CharDim))
            Vx_d = ustrip.(dimensionalize(Array(Vx_viz), cm / yr, CharDim))
            ∇V_d = ustrip.(dimensionalize(Array(∇V_viz), cm / yr, CharDim))
            P_d = ustrip.(dimensionalize(Array(P_viz), MPa, CharDim))
            ρg_d = ustrip.(dimensionalize(Array(ρg_viz), kg / m^3 * m / s^2, CharDim))
            ρ_d = ρg_d / 10
            ϕ_d = Array(ϕ_viz)
            τII_d = ustrip.(dimensionalize(Array(τII_viz), MPa, CharDim))
            τxy_d = ustrip.(dimensionalize(Array(τxy_viz), MPa, CharDim))
            εII_d = ustrip.(dimensionalize(Array(εII_viz), s^-1, CharDim))
            εxy_d = ustrip.(dimensionalize(Array(εxy_viz), s^-1, CharDim))
            t_yrs = dimensionalize(t, yr, CharDim)
            t_Kyrs = t_yrs / 1e3
            t_Myrs = t_Kyrs / 1e3

            p = particles.coords
            # pp = [argmax(p) for p in phase_ratios.center] #if you want to plot it in a heatmap rather than scatter
            ppx, ppy = p
            pxv = ustrip.(dimensionalize(ppx.data[:], km, CharDim))
            pyv = ustrip.(dimensionalize(ppy.data[:], km, CharDim))
            clr = pPhases.data[:]
            clrT = pT.data[:]
            idxv = particles.index.data[:]

            if igg.me == 0
                fig = Figure(; resolution=(2000, 2000), createmissing=true)
                ar = li[1] / li[2]
                # ar = DataAspect()

                ax0 = Axis(
                    fig[1, 1:2];
                    aspect=ar,
                    title="t = $((ustrip.(t_Kyrs))) Kyrs",
                    titlesize=50,
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

                ax1 = Axis(
                    fig[2, 1][1, 1];
                    aspect=ar,
                    title=L"T [\mathrm{C}]",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                # ax2 = Axis(fig[2,2][1,1], aspect = ar, title = L"\log_{10}(\eta [\mathrm{Pas}])",titlesize=40, yticklabelsize=25, xticklabelsize=25, xlabelsize=25)
                ax2 = Axis(
                    fig[2, 2][1, 1];
                    aspect=ar,
                    title=L"\log_{10}(\eta_{vep} [\mathrm{Pas}])",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                ax3 = Axis(
                    fig[3, 1][1, 1];
                    aspect=ar,
                    title=L"Vy [\mathrm{cm/yr}]",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                ax4 = Axis(
                    fig[3, 2][1, 1];
                    aspect=ar,
                    title=L"\log_{10}(\varepsilon_{\textrm{II}}[\mathrm{s}^{-1}])",
                    xlabel="Width [km]",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                ax5 = Axis(
                    fig[4, 1][1, 1];
                    aspect=ar,
                    title=L"Phases",
                    xlabel="Width [km]",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                ax6 = Axis(
                    fig[4, 2][1, 1];
                    aspect=ar,
                    title=L"\tau_{\textrm{II}} [MPa]",
                    xlabel="Width [km]",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )

                linkyaxes!(ax1, ax2)
                hideydecorations!(ax2; grid=false)
                linkyaxes!(ax3, ax4)
                hideydecorations!(ax4; grid=false)
                hidexdecorations!(ax1; grid=false)
                hidexdecorations!(ax3; grid=false)
                hidexdecorations!(ax2; grid=false)
                hidexdecorations!(ax4; grid=false)

                p1 = heatmap!(ax1, x_c, y_c, T_d; colormap=:batlow)
                contour!(ax1, x_c, y_c, T_d, ; color=:white, levels=600:100:900)
                p2 = heatmap!(ax2, x_c, y_c, log10.(η_vep_d); colormap=:glasgow) #, colorrange= (log10(minimum(η_vep_d)), log10(1e20)))
                p3 = heatmap!(ax3, x_v, y_v, Vy_d; colormap=:vik)
                p4 = heatmap!(ax4, x_v, y_v, log10.(εII_d); colormap=:glasgow)
                p5 = scatter!(
                    ax5, Array(pxv[idxv]), Array(pyv[idxv]); color=Array(clr[idxv])
                )
                arrows!(
                    ax5,
                    Xp_d[:],
                    Yp_d[:],
                    Vxp[:] * Vscale,
                    Vyp[:] * Vscale;
                    arrowsize=10,
                    lengthscale=70,
                    arrowcolor=:red,
                    linecolor=:red,
                )
                p6 = heatmap!(ax6, x_v, y_v, τII_d; colormap=:batlow)

                Colorbar(
                    fig[2, 1][1, 2], p1; height=Relative(0.7), ticklabelsize=25, ticksize=15
                )
                Colorbar(
                    fig[2, 2][1, 2], p2; height=Relative(0.7), ticklabelsize=25, ticksize=15
                )
                Colorbar(
                    fig[3, 1][1, 2], p3; height=Relative(0.7), ticklabelsize=25, ticksize=15
                )
                Colorbar(
                    fig[3, 2][1, 2], p4; height=Relative(0.7), ticklabelsize=25, ticksize=15
                )
                Colorbar(
                    fig[4, 1][1, 2], p5; height=Relative(0.7), ticklabelsize=25, ticksize=15
                )
                Colorbar(
                    fig[4, 2][1, 2], p6; height=Relative(0.7), ticklabelsize=25, ticksize=15
                )
                rowgap!(fig.layout, 1)
                colgap!(fig.layout, 1)
                colgap!(fig.layout, 1)
                colgap!(fig.layout, 1)
                fig
                figsave = joinpath(figdir, @sprintf("%06d.png", it))
                save(figsave, fig)

                let
                    Yv = [y for x in xvi[1], y in xvi[2]][:]
                    Y = [y for x in xci[1], y in xci[2]][:]
                    fig = Figure(; resolution=(1200, 900))
                    ax1 = Axis(fig[1, 1]; aspect=2 / 3, title="T")
                    ax2 = Axis(fig[1, 2]; aspect=2 / 3, title="Pressure")

                    scatter!(
                        ax1,
                        Array(ustrip.(dimensionalize(thermal.T[2:(end - 1), :][:], C, CharDim))),
                        ustrip.(dimensionalize(Yv, km, CharDim)),
                    )
                    lines!(
                        ax2,
                        Array(ustrip.(dimensionalize(stokes.P[:], MPa, CharDim))),
                        ustrip.(dimensionalize(Y, km, CharDim)),
                    )

                    hideydecorations!(ax2)
                    save(joinpath(figdir, "pressure_profile_$it.png"), fig)
                    fig
                end
            end
        end
    end
    # finalize_global_grid()

end


# function run()
    figname = "temp_debugging"
    # mkdir(figname)
    ar = 1 # aspect ratio
    n = 128
    nx = n * ar - 2
    ny = n - 2
    nz = n - 2
    igg = if !(JustRelax.MPI.Initialized())
        IGG(init_global_grid(nx, ny, 0; init_MPI=true)...)
    else
        igg
    end
    DikeInjection_2D(igg; figname=figname, nx=nx, ny=ny)

#     return figname, nx, ny, igg
# end

# # @time
# run()

function plot_particles(particles, pPhases)
    p = particles.coords
    # pp = [argmax(p) for p in phase_ratios.center] #if you want to plot it in a heatmap rather than scatter
    ppx, ppy = p
    # pxv = ustrip.(dimensionalize(ppx.data[:], km, CharDim))
    # pyv = ustrip.(dimensionalize(ppy.data[:], km, CharDim))
    pxv = ppx.data[:]
    pyv = ppy.data[:]
    clr = pPhases.data[:]
    # clrT = pT.data[:]
    idxv = particles.index.data[:]
    f,ax,h=scatter(Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), colormap=:roma)
    Colorbar(f[1,2], h)
    f
end
