using CUDA
CUDA.allowscalar(true)
using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
const backend_JR = CUDABackend

using ParallelStencil, ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(CUDA, Float64, 2) #or (CUDA, Float64, 2) or (AMDGPU, Float64, 2)

using JustPIC
using JustPIC._2D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend = CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

# Load script dependencies
using Printf, Statistics, LinearAlgebra, GeoParams, CairoMakie, CellArrays
import GeoParams.Dislocation
using StaticArrays
# using ParallelStencil.FiniteDifferences2D   #specify Dimension
# using ImplicitGlobalGrid
# using MPI: MPI
using GeophysicalModelGenerator#, StaticArrays
using WriteVTK

# -----------------------------------------------------------------------------------------
# Viscosity calculation functions stored in this script
include("./src/Particles_Helperfunc.jl");
# include("./src/Helperfunc_old.jl");
include("./src/LoadModel.jl");
println("Loaded helper functions")
#------------------------------------------------------------------------------------------
LonLat = load("./Data/ExportToba_2.jld2", "TobaTopography_LATLON");

proj = ProjectionPoint(; Lat=2.19, Lon=98.91);
Topo = convert2CartData(LonLat, proj);

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

            if y < 0.0 && phases_topo[i, j] > 1.0
                @cell phases[ip, i, j] = 4.0
            end

        end
        return nothing
    end

    @parallel (@idx ni) init_phases!(
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
            depth_corrected[i, j] = -depth_corrected[i, j]
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

function BC_topography(Vx,Vy, εbg, depth_corrected, xvi, lx,ly)
    xv, yv = xvi

    @parallel_indices (i, j) function pure_shear_x!(Vx)
        xi = xv[i]
        Vx[i, j + 1] = εbg * (xi - lx * 0.5) / (lx/2)/2
        return nothing
    end

    @parallel_indices (i, j) function pure_shear_y!(Vy)
        yi = max(depth_corrected[i,j],0.0)
        Vy[i + 1, j] = abs(yi) * εbg / ly
        return nothing
    end

    nx, ny = size(Vx)
    @parallel (1:nx, 1:(ny - 2)) pure_shear_x!(Vx)
    nx, ny = size(Vy)
    @parallel (1:(nx - 2), 1:ny) pure_shear_y!(Vy)

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

# #Multiplies parameter with the fraction of a phase
# @generated function compute_param_times_frac(
#     fn::F,
#     PhaseRatios::Union{NTuple{N,T},SVector{N,T}},
#     MatParam::NTuple{N,AbstractMaterialParamsStruct},
#     argsi,
#     ) where {F,N,T}
#     # # Unrolled dot product
#     quote
#         val = zero($T)
#         Base.Cartesian.@nexprs $N i ->
#         val += @inbounds PhaseRatios[i] * fn(MatParam[i], argsi)
#         return val
#     end
# end


@parallel_indices (I...) function compute_melt_fraction!(ϕ, phase_ratios, rheology, args)
    ϕ[I...] = compute_melt_frac(rheology, (;T=args.T[I...]), phase_ratios[I...])
    return nothing
end

@inline function compute_melt_frac(rheology, args, phase_ratios)
    return GeoParams.compute_meltfraction_ratio(phase_ratios, rheology, args)
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

    @parallel (@idx ni) _phase_change!( phases, particles.coords..., particles.index)
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

    @parallel (@idx ni) _open_conduit!(phases, particles.coords..., particles.index, xc_conduit, yc_conduit, r_conduit)
end

function new_thermal_anomaly(phases, particles, xc_anomaly, yc_anomaly, r_anomaly)
    ni = size(phases)

    @parallel_indices (I...) function new_anomlay_particles(phases, px, py, index, xc_anomaly, yc_anomaly, r_anomaly)
        @inbounds for ip in JustRelax.cellaxes(phases)
            @cell(index[ip, I...]) == 0 && continue

            x = JustRelax.@cell px[ip, I...]
            y = -(JustRelax.@cell py[ip, I...])

            # thermal anomaly - circular
            if ((x - xc_anomaly)^2 + (y + yc_anomaly)^2 ≤ r_anomaly^2)
                JustRelax.@cell phases[ip, I...] = 3.0
            end
        end
        return nothing
    end
    @parallel (@idx ni) new_anomlay_particles(phases, particles.coords..., particles.index, xc_anomaly, yc_anomaly, r_anomaly)
end

function Caldera_2D(igg; figname=figname, nx=nx, ny=ny, do_vtk=false)

    #-----------------------------------------------------
    # USER INPUTS
    #-----------------------------------------------------
    Topography= true; #specify if you want topography plotted in the figures
    Freesurface = true #specify if you want to use freesurface
        sticky_air = 5km #specify the thickness of the sticky air layer
    Inject_Dike = false #specify if you want to inject a dike
    dike_width = 2.0km  #specify the width of the dike
    dike_height = 0.50km #specify the height of the dike
    dike_temp = 1000C   #specify the temperature of the dike
    InjectionInterval = 25yr        # number of timesteps between injections (if Inject_Dike=true). Increase this at higher resolution.
    #now injecting every 2 it

    toy = true #specify if you want to use the toy model or the Toba model
    thermal_perturbation = :elliptical_anomaly #specify if you want a thermal perturbation in the center of the domain or random
    sphere = 15km                               #specify the radius of the circular anomaly
    ellipse = 15km
    temp_anomaly = 1000C

    shear = true #specify if you want to use pure shear boundary conditions
    # v_extension = 1e-10m/s
    # regime = ViscoElastic                  #Rheology of the Stokes Solver: ViscoElastic, ViscoElastoPlastic

    Nx, Ny, Nz = 100, 100, 100      # 3D grid size (does not yet matter)
    arrow_steps_x_dir = 5           # number of arrows in the quiver plot
    arrow_steps_y_dir = 4           # number of arrows in the quiver plot

    nt = 500                        # number of timesteps

    η_uppercrust = 1e21             #viscosity of the upper crust
    η_magma = 1e16                  #viscosity of the magma
    η_air = 1e16                    #viscosity of the air

    #-----------------------------------------------------

    # IO ------------------------------------------------
    # IO ----- -------------------------------------------
    # if it does not exist, make folder where figures are stored
    figdir = "./fig2D/$figname/"
    if do_vtk
        vtk_dir      = joinpath(figdir,"vtk")
        take(vtk_dir)
    end
    take(figdir)
    # ----------------------------------------------------

    # Standard MTK Toba routine--------------------------------------------------------------

    # nondimensionalize with CharDim
    CharDim = GEO_units(; length=40km, viscosity=1e20Pa * s)
    ni = (nx, ny)   #grid spacing for calculation (without ghost cells)

    Grid = create_CartGrid(;
        size=(Nx, Ny, Nz),
        x=((Topo.x.val[1, 1, 1])km, (Topo.x.val[end, 1, 1])km),
        y=((Topo.y.val[1, 1, 1])km, (Topo.y.val[1, end, 1])km),
        z=(-40km, 4km),
    )
    X, Y, Z = xyz_grid(Grid.coord1D...)
    DataTopo = CartData(X, Y, Z, (Depthdata=Z,))

    Lon, Lat, Depth = (Topo.x.val .* km), (Topo.y.val .* km), ((Topo.z.val ./ 1e3) .* km)
    Topo_Cart = CartData(Lon, Lat, Depth, (Depth=Depth,))

    ind = above_surface(DataTopo, Topo_Cart)
    Phase = ones(size(X))
    Phase[ind] .= 3

    DataPara = CartData(X, Y, Z, (Phase=Phase,))
    Phase = Int64.(round.(DataPara.fields.Phase))

    # ----------CrossSections for 2D simulation----------------------------------


    Data_Cross = cross_section(
        DataPara;
        dims=(ni[1] + 1, ni[2] + 1),
        Interpolate=true,
        Start=(ustrip(-40.00km), ustrip(50.00km)),
        End=(ustrip(-06.00km), ustrip(20.00km)),
    )

    x_new = flatten_cross_section(Data_Cross)
    Data_Cross = addfield(Data_Cross, "FlatCrossSection", x_new)
    Phase = dropdims(Data_Cross.fields.Phase; dims=3) #Dropped the 3rd dim of the Phases to be consistent with new grid of the CrossSection
    Phase = Int64.(round.(Phase))
    Phase[Phase .== 2] .= 3 #set the phase of the air to 3

    Topo_Cross = cross_section(
        Topo_Cart;
        dims=(ni[1] + 1, ni[2] + 1),
        Interpolate=true,
        Start=(ustrip(-40.00km), ustrip(50.00km)),
        End=(ustrip(-06.00km), ustrip(20.00km)),
    )

    x_new_Topo = flatten_cross_section(Topo_Cross)
    Topo_Cross = addfield(Topo_Cross, "FlatCrossSection", x_new_Topo)
    Topo_nondim = nondimensionalize(Topo_Cross.fields.Depth, CharDim)

    ## Workaround to display topo on heatmaps for η,ρ,ϕ due to different dimensions (ni, not ni+1)
    Data_Cross_ni = cross_section(
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
    Model3D_Cross = cross_section(
        Model3D_cart;
        dims=(ni[1] + 1, ni[2] + 1),
        Interpolate=true,
        Start=(ustrip(-40.00km), ustrip(50.00km)),
        End=(ustrip(-06.00km), ustrip(20.00km)),
    )

    Model3D_new = flatten_cross_section(Model3D_Cross)
    Model3D_Cross = addfield(Model3D_Cross, "Model3D_Cross", Model3D_new)

    #seismo model with ni dimensions
    Model3D_Cross_ni = cross_section(
        Model3D_cart;
        dims=(ni[1], ni[2]),
        Interpolate=true,
        Start=(ustrip(-40.00km), ustrip(50.00km)),
        End=(ustrip(-06.00km), ustrip(20.00km)),
    )

    Model3D_new_ni = flatten_cross_section(Model3D_Cross_ni)
    Model3D_Cross_ni = addfield(Model3D_Cross_ni, "Model3D_Cross", Model3D_new_ni)

    #New 2D Grid
    if Topography == true
        Grid2D = create_CartGrid(;
            size=(ni[1], ni[2]),
            x=(extrema(Data_Cross.fields.FlatCrossSection) .* km),
            z=(
                # (minimum(Data_Cross.z.val) ./ 2.0) .* km, (maximum(Data_Cross.z.val)) .* km
                (-25.0) .* km, (maximum(Data_Cross.z.val)) .* km
            ),
            CharDim=CharDim,
        ) #create new 2D grid for Injection routine
    elseif Freesurface == true
        # new grid with no topography as the topo now seems to be the problem with strain rate.

        Grid2D = create_CartGrid(;
            size=(ni[1], ni[2]),
            x=(extrema(Data_Cross.fields.FlatCrossSection) .* km),
            z=((-25.0) .* km, sticky_air),
            CharDim=CharDim,
        ) #create new 2D grid for Injection routine
    else
        # new grid with no topography as the topo now seems to be the problem with strain rate.

        Grid2D = create_CartGrid(;
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
    η_reg   = 1.0e16Pas           # regularisation "viscosity" for Drucker-Prager
    Coh     = 10.0MPa              # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ       = 30.0 * do_DP         # friction angle
    G0      = 25e9Pa        # elastic shear modulus
    G_magma = 10e9Pa        # elastic shear modulus perturbation
    εbg     = 2e-14 / s             # background strain rate
    εbg     = nondimensionalize(εbg, CharDim) # background strain rate

    # soft_C      = LinearSoftening((ustrip(Coh)/2, ustrip(Coh)), (0e0, 1e-1)) # softening law
    soft_C      =  soft_C  = NonLinearSoftening(;ξ₀ = ustrip.(Coh), Δ=ustrip.(Coh)/99)
    pl          = DruckerPrager_regularised(; C=Coh, ϕ=ϕ, η_vp=η_reg, Ψ=0.0, softening_C = soft_C)        # plasticity

    # el       = SetConstantElasticity(; G=G0, ν=0.46)                            # elastic spring
    el       = SetConstantElasticity(; G=G0, ν=0.3)                            # elastic spring
    el_magma = SetConstantElasticity(; G=G_magma, ν=0.3)                            # elastic spring
    el_air   = SetConstantElasticity(; ν=0.3, Kb=0.101MPa)                            # elastic spring
    # el_air   = SetConstantElasticity(; G=G0, ν=0.3)                            # elastic spring
    disl_upper_crust = DislocationCreep(;
        A=5.07e-18, n=2.3, E=154e3, V=0.0, r=0.0, R=8.3145
    ) #(;E=187kJ/mol,) Kiss et al. 2023
    # creep_rock = LinearViscous(; η=η_uppercrust * Pa * s)
    creep_rock = SetDislocationCreep(Dislocation.wet_quartzite_Hirth_2001)
    # creep_rock = DislocationCreep(; A=1.67e-24, n=3.5, E=1.87e5, V=6e-6, r=0.0, R=8.3145)
    creep_magma = LinearViscous(; η=η_magma * Pa * s)
    creep_air = LinearViscous(; η=η_air * Pa * s)
    cutoff_visc = (
        nondimensionalize(1e16Pa * s, CharDim), nondimensionalize(1e24Pa * s, CharDim)
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
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    #---------------------------------------------------------------------------------------

    # Set material parameters
    MatParam = (
        #Name="UpperCrust"
        SetMaterialParams(;
            Phase   = 1,
            Density  = PT_Density(ρ0=2700kg/m^3, β=β_rock/Pa),
            HeatCapacity = ConstantHeatCapacity(Cp=1050J/kg/K),
            Conductivity = ConstantConductivity(k=3.0Watt/K/m),
            LatentHeat = ConstantLatentHeat(Q_L=350e3J/kg),
            ShearHeat         = ConstantShearheating(1.0NoUnits),
            CompositeRheology = CompositeRheology((creep_rock, el, pl, )),
            Melting = MeltingParam_Smooth3rdOrder(),
            Elasticity = el,
            CharDim  = CharDim,),

        #Name="Magma"
        SetMaterialParams(;
            Phase   = 2,
            Density  = PT_Density(ρ0=2600kg/m^3, β=β_magma/Pa),
            HeatCapacity = ConstantHeatCapacity(Cp=1050J/kg/K),
            Conductivity = ConstantConductivity(k=1.5Watt/K/m),
            LatentHeat = ConstantLatentHeat(Q_L=350e3J/kg),
            ShearHeat         = ConstantShearheating(0.0NoUnits),
            CompositeRheology = CompositeRheology((creep_magma, el_magma)),
            Melting = MeltingParam_Smooth3rdOrder(),
            Elasticity = el_magma,
            CharDim  = CharDim,),

        #Name="Thermal Anomaly"
        SetMaterialParams(;
            Phase   = 3,
            Density  = PT_Density(ρ0=2600kg/m^3, β=β_magma/Pa),
            HeatCapacity = ConstantHeatCapacity(Cp=1050J/kg/K),
            Conductivity = ConstantConductivity(k=1.5Watt/K/m),
            LatentHeat = ConstantLatentHeat(Q_L=350e3J/kg),
            ShearHeat         = ConstantShearheating(0.0NoUnits),
            CompositeRheology = CompositeRheology((creep_magma,el_magma)),
            Melting = MeltingParam_Smooth3rdOrder(),
            Elasticity = el_magma,
            CharDim  = CharDim,),

        #Name="Sticky Air"
        SetMaterialParams(;
            Phase   = 4,
            Density   = ConstantDensity(ρ=10kg/m^3,),
            HeatCapacity = ConstantHeatCapacity(Cp=1000J/kg/K),
            Conductivity = ConstantConductivity(k=15Watt/K/m),
            LatentHeat = ConstantLatentHeat(Q_L=0.0J/kg),
            ShearHeat         = ConstantShearheating(0.0NoUnits),
            # CompositeRheology = CompositeRheology((creep_air,)),
            CompositeRheology = CompositeRheology((creep_air,el)),
            Elasticity = el,
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

    depth_corrected_v = PTArray(backend_JR)(depth_corrected_v)
    depth_corrected_c = PTArray(backend_JR)(depth_corrected_c)
    phases_topo_v = (PTArray(backend_JR)(phases_topo_v))
    phases_topo_c = (PTArray(backend_JR)(phases_topo_c))
    vertex2center!(phases_topo_c, phases_topo_v)
    vertex2center!(depth_corrected_c, depth_corrected_v)
    @views phases_topo_c = round.(phases_topo_c)
    @views phases_topo_c[phases_topo_c .== 2.0] .= 1.0
    @views phases_topo_c[phases_topo_c .== 3.0] .= 4.0

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 30, 40, 20 # nxcell = initial number of particles per cell; max_cell = maximum particles accepted in a cell; min_xcell = minimum number of particles in a cell
    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...
    )

    subgrid_arrays   = SubgridDiffusionCellArrays(particles)

    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pT, pT0, pPhases    = init_cell_arrays(particles, Val(3))
    particle_args       = (pT, pT0, pPhases)

    if Topography == true
        xc, yc = 0.5 * lx, 0.17 * -lz  # origin of thermal anomaly
        x_anomaly, y_anomaly = lx * 0.5, -lz * 0.17  # Randomly vary center of dike
        # coordinates for case: ellipse = 35km
        # xc, yc = 0.66 * lx, 0.4 * -lz  # origin of thermal anomaly
        # x_anomaly, y_anomaly = lx * 0.66, -lz * 0.5  # Randomly vary center of dike
    elseif Freesurface == true
        xc, yc = 0.5 * lx, 0.17 * -lz  # origin of thermal anomaly
        x_anomaly, y_anomaly = lx * 0.5, -lz * 0.17  # Randomly vary center of dike
        # xc, yc = 0.5 * lx, 0.5 * -(lz-nondimensionalize(sticky_air,CharDim))  # origin of thermal anomaly
        # x_anomaly, y_anomaly = lx * 0.5, -(lz-nondimensionalize(sticky_air,CharDim)) * 0.6 # Randomly vary center of dike
    else
        xc, yc = 0.5 * lx, 0.5 * -lz # origin of thermal anomaly
        x_anomaly, y_anomaly = lx * 0.5, -lz * 0.5 # Randomly vary center of dike
    end

    radius = nondimensionalize(ellipse, CharDim)         # radius of perturbation
    a = nondimensionalize(15km, CharDim)
    b = nondimensionalize(5km, CharDim)
    r_anomaly = nondimensionalize(0.75km,CharDim)             # radius of perturbation
    # Case for ellipse = 35km
    # r_anomaly = nondimensionalize(1.5km,CharDim)             # radius of perturbation
    xc_conduit, yc_conduit, r_conduit = (lx*0.5),nondimensionalize(4.15km,CharDim),nondimensionalize(2.3km,CharDim)

    init_phases!(pPhases, particles, phases_topo_v, xc, yc, a, b, radius, x_anomaly, y_anomaly, r_anomaly, xc_conduit, yc_conduit, r_conduit)
    phase_ratios = PhaseRatio(backend_JR, ni, length(MatParam))

    phase_ratios_center!(phase_ratios, particles, grid, pPhases)

    # Physical Parameters
    geotherm = GeoUnit(30K / km)
    geotherm_nd = ustrip(Value(nondimensionalize(geotherm, CharDim)))
    ΔT = geotherm_nd * (lz - nondimensionalize(sticky_air,CharDim)) # temperature difference between top and bottom of the domain
    tempoffset = nondimensionalize(0C, CharDim)
    η = MatParam[2].CompositeRheology[1][1].η.val       # viscosity for the Rayleigh number
    Cp0 = MatParam[2].HeatCapacity[1].Cp.val              # heat capacity
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
    stokes          = StokesArrays(backend_JR, ni)
    pt_stokes       = PTStokesCoeffs(li, di; ϵ=1e-4, CFL=0.99 / √2.1) #ϵ=1e-4,  CFL=1 / √2.1 CFL=0.27 / √2.1

    thermal         = ThermalArrays(backend_JR, ni)
    thermal_bc      = TemperatureBoundaryConditions(;
        no_flux     = (left = true, right = true, top = false, bot = false),
    )

    if Topography == true
        depth_corrected_dim = ustrip.(dimensionalize(depth_corrected_v, km, CharDim))
    else
        xyv = PTArray(backend_JR)([y for x in xvi[1], y in xvi[2]])
        depth_corrected_dim = ustrip.(dimensionalize(-xyv, km, CharDim))
    end

    @parallel (@idx ni .+ 1) init_T!(
        thermal.T, phases_topo_v, depth_corrected_v, geotherm_nd,tempoffset
    )

    thermal_bcs!(thermal.T, thermal_bc)
    temperature2center!(thermal)

    Tnew_cpu = Array{Float64}(undef, ni .+ 1...)                  # Temperature for the CPU
    Phi_melt_cpu = Array{Float64}(undef, ni...)                    # Melt fraction for the CPU

    args = (; T=thermal.Tc, P=stokes.P, dt=dt,  ΔTc=thermal.ΔTc)

    # PT coefficients for thermal diffusion -------------
    pt_thermal = PTThermalCoeffs(
        backend_JR, MatParam, phase_ratios, args, dt, ni, di, li; ϵ=1e-5, CFL=0.8 / √2.1
    )
    # Boundary conditions of the flow
    if shear == true
        BC_topography(@velocity(stokes)..., εbg, depth_corrected_v,xvi,lx,lz)
    end
    flow_bcs = FlowBoundaryConditions(;
        free_slip=(left=true, right=true, top=true, bot=true),
        free_surface =true,
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(stokes.V.Vx, stokes.V.Vy)

    # G = @fill(MatParam[1].Elasticity[1].G.val, ni...)     # initialise shear modulus
    ϕ = @zeros(ni...)                                        # melt fraction center

    # Buoyancy force
    ρg = @zeros(ni...), @zeros(ni...)                      # ρg[1] is the buoyancy force in the x direction, ρg[2] is the buoyancy force in the y direction

    # Preparation for Visualisation
    ni_v_viz  = nx_v_viz, ny_v_viz = (ni[1] - 1) * igg.dims[1], (ni[2] - 1) * igg.dims[2]      # size of the visualisation grid on the vertices according to MPI dims
    ni_viz    = nx_viz, ny_viz = (ni[1] - 2) * igg.dims[1], (ni[2] - 2) * igg.dims[2]            # size of the visualisation grid on the vertices according to MPI dims
    Vx_vertex = PTArray(backend_JR)(ones(ni .+ 1...))                                                  # initialise velocity for the vertices in x direction
    Vy_vertex = PTArray(backend_JR)(ones(ni .+ 1...))                                                  # initialise velocity for the vertices in y direction

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

    # Dike parameters for injection
    W_in, H_in              = nondimensionalize(dike_width, CharDim),nondimensionalize(dike_height, CharDim); # Width and thickness of dike
    T_in                    = nondimensionalize(dike_temp, CharDim);
    # H_ran, W_ran            = length(z1) ,length(x1);                          # Size of domain in which we randomly place dikes and range of angles
    # H_ran, W_ran            = (xvi[1].stop, xvi[2].start).* [0.3;0.4];                          # Size of domain in which we randomly place dikes and range of angles
    H_ran, W_ran            = (xvi[1].stop, xvi[2].start).* [0.2;0.3];                          # Size of domain in which we randomly place dikes and range of angles
    Dike_Type               = "ElasticDike"
    # Tracers                 = StructArray{Tracer}(undef, 1);
    nTr_dike                = 300;

    # Arguments for functions
    args = (; ϕ=ϕ, T=thermal.Tc, P=stokes.P, dt=dt, ΔTc=thermal.ΔTc)
    @copy thermal.Told thermal.T
    @copy Tnew_cpu Array(thermal.T[2:(end - 1), :])

    for _ in 1:5
        compute_ρg!(ρg[2], phase_ratios, MatParam, (T=thermal.Tc, P=stokes.P))
        @parallel (@idx ni) init_P!(stokes.P, ρg[2], phases_topo_c, depth_corrected_c)
    end

    compute_viscosity!(stokes, phase_ratios, args, MatParam, cutoff_visc)

    # pressure_top = nondimensionalize(0.0MPa,CharDim)
    # temp_sticky_air = nondimensionalize(0.0C,CharDim)
    if toy == true
        if thermal_perturbation == :random
            δT = 5.0              # thermal perturbation (in %)
            random_perturbation!(
                thermal.T, δT, (lx * 1 / 8, lx * 7 / 8), (-2000e3, -2600e3), xvi
            )
        elseif thermal_perturbation == :circular
            δT = 20.0              # thermal perturbation (in %)
            r  = nondimensionalize(5km, CharDim)         # radius of perturbation
            circular_perturbation!(thermal.T, δT, xc, yc, r, xvi)

        elseif thermal_perturbation == :circular_anomaly
            anomaly = nondimensionalize(temp_anomaly, CharDim) # temperature anomaly
            radius  = nondimensionalize(sphere, CharDim)         # radius of perturbation
            circular_anomaly!(thermal.T, anomaly, xc, yc, radius, xvi)

        elseif thermal_perturbation == :elliptical_anomaly
            anomaly = nondimensionalize(temp_anomaly, CharDim) # temperature anomaly
            radius  = nondimensionalize(ellipse, CharDim)         # radius of perturbation
            offset  = nondimensionalize(500C, CharDim)
            δT      = 25.0              # thermal perturbation (in %)
            elliptical_anomaly_gradient!(
                    thermal.T, offset, xc, yc, a, b, radius, xvi
                    )
            # conduit_gradient!(thermal.T, offset, xc_conduit, -yc_conduit, r_conduit, xvi)
            circular_perturbation!(thermal.T, δT, x_anomaly, -y_anomaly, r_anomaly, xvi)
        end
    end
    # make sure they are the same
    thermal.Told .= thermal.T
    temperature2center!(thermal)
    @parallel (@idx ni) compute_melt_fraction!(
        ϕ, phase_ratios.center, MatParam, (T=thermal.Tc, P=stokes.P)
    )

    # Time loop
    t, it      = 0.0, 0
    interval   = 1.0
    InjectVol  = 0.0
    # evo_t      = Float64[]
    # evo_InjVol = Float64[]
    local Vx_v, Vy_v
    local iters
    if do_vtk
        Vx_v = @zeros(ni.+1...)
        Vy_v = @zeros(ni.+1...)
    end

    T_buffer    = @zeros(ni.+1)
    Told_buffer = similar(T_buffer)
    Tsurf  = thermal.T[1, end]
    Tbot   = thermal.T[1, 1]
    dt₀         = similar(stokes.P)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end
    grid2particle!(pT, xvi, T_buffer, particles)
    @copy stokes.P0 stokes.P

    # Plot initial T and η profiles
    let
        Yv = [y for x in xvi[1], y in xvi[2]][:]
        Y = [y for x in xci[1], y in xci[2]][:]
        fig = Figure(; size=(1200, 900))
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

    # dt *= 0.2
    while it < 150 #nt


        if rem(it, 25) == 0
            # if it > 75 && rem(it, 75) == 0
            x_anomaly, y_anomaly = lx * 0.5, -lz * 0.17  # Randomly vary center of dike
                r_anomaly = nondimensionalize(0.75km,CharDim)
                δT = 30.0              # thermal perturbation (in %)
                new_thermal_anomaly(pPhases, particles, x_anomaly, y_anomaly, r_anomaly)
                circular_perturbation!(thermal.T, δT, x_anomaly, -y_anomaly, r_anomaly, xvi)
                for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
                    copyinn_x!(dst, src)
                end
                @views T_buffer[:, end] .= nondimensionalize(0.0C, CharDim)
                @views thermal.T[2:end-1, :] .= T_buffer
                temperature2center!(thermal)
                grid2particle_flip!(pT, xvi, T_buffer, Told_buffer, particles)

                phase_ratios_center!(phase_ratios, particles, grid, pPhases)
            end

        args = (; ϕ=ϕ, T=thermal.Tc, P=stokes.P, dt=dt, ΔTc=thermal.ΔTc)

        if Inject_Dike == true
            # if rem(it, 2) == 0
          # if  ustrip(dimensionalize(t,yr,CharDim)) >= (ustrip(InjectionInterval)*interval) #Inject every time the physical time exceeds the injection interval
            #   ID             =     rand(ind_melt)
            #   cen            = [x1[ID],z1[ID]]  # Randomly vary center of dike
              cen              = ((xvi[1].stop, xvi[2].stop).+(xvi[1].start,xvi[2].start))./2.0 .+ rand(-0.5:1e-3:0.5, 2).*[W_ran;H_ran];
              #   Dike injection based on Toba seismic inversion
              if cen[end] < ustrip(nondimensionalize(-5km, CharDim))
                  Angle_rand = [rand(80.0:0.1:100.0)] # Orientation: near-vertical @ depth
              else
                  Angle_rand = [rand(-10.0:0.1:10.0)] # Orientation: near-vertical @ shallower depth
              end

              dike      =   Dike(Angle=Angle_rand, W=W_in, H=H_in, Type=Dike_Type, T=T_in, Center=cen, Phase =2 );

            Inject_dike_particles2D!(pPhases,pT, particles, dike)
            particle2grid!(T_buffer, pT, xvi, particles)
            @views T_buffer[:, end] .= Tsurf
            @views T_buffer[:, 1] .= Tbot
            @views thermal.T[2:end - 1, :] .= T_buffer
            thermal_bcs!(thermal.T, thermal_bc)
            temperature2center!(thermal)
            thermal.ΔT .= thermal.T .- thermal.Told
            vertex2center!(thermal.ΔTc, thermal.ΔT)

            phase_ratios_center(phase_ratios, particles, grid, pPhases)

            @parallel (@idx ni) compute_melt_fraction!(
                ϕ, phase_ratios.center, MatParam, (T=thermal.Tc, P=stokes.P)
            )

        #   end
        end
        # #open the conduit
        # if it == 114
        #     conduit_gradient_TBuffer!(T_buffer, offset, xc_conduit, -yc_conduit, r_conduit, xvi)
        #     open_conduit!(pPhases, particles, xc_conduit, yc_conduit, r_conduit)
        #     grid2particle_flip!(pT, xvi, T_buffer, Told_buffer, particles.coords)
        # end

        # @views T_buffer[:, end] .= nondimensionalize(0.0C, CharDim)
        # @views T_buffer[args.sticky_air.==4.0] .= nondimensionalize(0.0C, CharDim)
        # # @views T_buffer[:, 1] .= maximum(thermal.T)
        # @views thermal.T[2:end-1, :] .= T_buffer
        # temperature2center!(thermal)

        args = (; ϕ=ϕ, T=thermal.Tc, P=stokes.P, dt=dt)
        compute_ρg!(ρg[end], phase_ratios, MatParam, (T=thermal.Tc, P=stokes.P))
        compute_viscosity!(
            stokes, phase_ratios, args, MatParam, cutoff_visc
        )
        args = (; ϕ=ϕ, T=thermal.Tc, P=stokes.P, dt=dt, ΔTc=thermal.ΔTc)
        # Stokes solver -----------------
        solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            phase_ratios,
            MatParam,
            args,
            dt,
            igg;
            kwargs = (;
                iterMax = 200e3,
                nout=1e3,
                viscosity_cutoff=cutoff_visc,
            )
        )
        tensor_invariant!(stokes.ε)

        dt = compute_dt(stokes, di, dt_diff, igg) #* 0.1
        # ------------------------------
        @show dt

        compute_shear_heating!(
            thermal,
            stokes,
            phase_ratios,
            MatParam, # needs to be a tuple
            dt,
        )
        # Thermal solver ---------------
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            MatParam,
            args,
            dt,
            di;
            kwargs = (;
                igg     = igg,
                phase   = phase_ratios,
                iterMax = 10e3,
                nout    = 1e2,
                verbose = true,
            )
        )

        for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
            copyinn_x!(dst, src)
        end
        subgrid_characteristic_time!(
            subgrid_arrays, particles, dt₀, phase_ratios, MatParam, thermal, stokes, xci, di
        )
        centroid2particle!(subgrid_arrays.dt₀, xci, dt₀, particles)
        subgrid_diffusion!(
            pT, T_buffer, thermal.ΔT[2:end-1, :], subgrid_arrays, particles, xvi,  di, dt
        )
        # ------------------------------

        # Advection --------------------
        # advect particles in space
              # Advection --------------------
        # advect particles in space
        advection!(particles, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy), dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (pT, ), (T_buffer, ), xvi)
        #phase change for particles
        phase_change!(pPhases, particles)
        # update phase ratios
        phase_ratios_center!(phase_ratios, particles, grid, pPhases)
        @parallel (@idx ni) compute_melt_fraction!(
            ϕ, phase_ratios.center, MatParam, (T=thermal.Tc, P=stokes.P)
        )

        particle2grid!(T_buffer, pT, xvi, particles)
        @views T_buffer[:, end] .= Tsurf
        @views T_buffer[:, 1] .= Tbot
        @views thermal.T[2:end - 1, :] .= T_buffer
        thermal_bcs!(thermal.T, thermal_bc)
        temperature2center!(thermal)
        thermal.ΔT .= thermal.T .- thermal.Told
        vertex2center!(thermal.ΔTc, thermal.ΔT)

        @show it += 1
        t += dt

        #  # # Plotting -------------------------------------------------------
        if it == 1 || rem(it, 1) == 0
            # checkpointing_jld2(figdir, stokes, thermal, η, particles, pPhases, t; igg=igg)

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
            # Vscale = 0.5 / maximum(sqrt.(Vxp .^ 2 + Vyp .^ 2)) * di[1] * (st_x - 1)

            velocity2vertex!(Vx_vertex, Vy_vertex, stokes.V.Vx, stokes.V.Vy)
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
            η_inn = Array(stokes.viscosity.η[2:(end - 1), 2:(end - 1)])
            η_vep_inn = Array(stokes.viscosity.η_vep[2:(end - 1), 2:(end - 1)])
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

            if do_vtk
                # JustRelax.velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)

                data_v = (;
                    T   = Array(T_d),
                    τxy = Array(τxy_d),
                    εxy = Array(εxy_d),
                    Vx  = Array(Vx_d),
                    Vy  = Array(Vy_d),
                )
                data_c = (;
                    P   = Array(P_d),
                    τII = Array(τII_d),
                    η   = Array(η_d),
                    ϕ   = Array(ϕ_d),
                    ρ  = Array(ρ_d),
                )
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                    (x_v,y_v),
                    (x_c,y_c),
                    data_v,
                    data_c,
                )
            end

            if igg.me == 0
                fig = Figure(; size=(2000, 1800), createmissing=true)
                ar = li[1] / li[2]
                # ar = DataAspect()

                ax0 = Axis(
                    fig[1, 1:2];
                    aspect=ar,
                    title="t = $(round.(ustrip.(t_Kyrs); digits=3)) Kyrs",
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
                ax2 = Axis(
                    fig[2, 2][1, 1];
                    aspect=ar,
                    title=L"\log_{10}(\eta_{vep}) [\mathrm{Pas}]",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                ax3 = Axis(
                    fig[3, 1][1, 1];
                    aspect=ar,
                    title=L"Vy [\mathrm{cm/yr}]",
                    # title=L"\tau_{\textrm{II}} [MPa]",
                    xlabel="Width [km]",
                    titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,
                )
                ax4 = Axis(
                    fig[3, 2][1, 1];
                    aspect=ar,
                    title=L"\log_{10}(\dot{\varepsilon}_{\textrm{II}}) [\mathrm{s}^{-1}]",
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
                linkyaxes!(ax3, ax4)
                hidexdecorations!(ax1; grid=false)
                hideydecorations!(ax2; grid=false)
                # hidexdecorations!(ax3; grid=false)
                # hidexdecorations!(ax2; grid=false)
                # hidexdecorations!(ax4; grid=false)
                pp = [argmax(p) for p in phase_ratios.center];
                @views pp = pp[2:end-1,2:end-1]
                @views T_d[pp.==4.0] .= NaN
                @views η_vep_d[pp.==4.0] .= NaN
                @views τII_d[pp.==4.0] .= NaN
                @views εII_d[pp.==4.0] .= NaN
                @views ϕ_d[pp.==4.0] .= NaN
                @views Vy_d[1:end-1, 1:end-1][pp.==4.0] .=NaN

                p1 = heatmap!(ax1, x_c, y_c, T_d; colormap=:batlow, colorrange=(000, 1200))
                contour!(ax1, x_c, y_c, T_d, ; color=:white, levels=600:200:1200)
                p2 = heatmap!(ax2, x_c, y_c, log10.(η_vep_d); colormap=:glasgow, colorrange= (log10(1e16), log10(1e22)))
                contour!(ax2, x_c, y_c, T_d, ; color=:white, levels=600:200:1200, labels = true)
                p3 = heatmap!(ax3, x_v, y_v, Vy_d; colormap=:vik)
                # p3 = heatmap!(ax3, x_v, y_v, τII_d; colormap=:batlow)
                p4 = heatmap!(ax4, x_c, y_c, log10.(εII_d); colormap=:glasgow, colorrange= (log10(5e-15), log10(5e-12)))
                p5 = scatter!(
                    ax5, Array(pxv[idxv]), Array(pyv[idxv]); color=Array(clr[idxv]), markersize=2
                )
                arrows!(
                    ax5,
                    x_c[1:5:end-1], y_c[1:5:end-1], Array.((Vx_d[1:5:end-1, 1:5:end-1], Vy_d[1:5:end-1, 1:5:end-1]))...,
                    lengthscale = 5 / max(maximum(Vx_d),  maximum(Vy_d)),
                    color = :red,
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
                    fig = Figure(; size=(1200, 900))
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
    figname = "Caldera_setup_V1"
    # mkdir(figname)
    do_vtk = true
    ar = 2 # aspect ratio
    n = 128
    nx = n * ar - 2
    ny = n - 2
    nz = n - 2
    igg = if !(JustRelax.MPI.Initialized())
        IGG(init_global_grid(nx, ny, 1; init_MPI=true)...)
    else
        igg
    end
    Caldera_2D(igg; figname=figname, nx=nx, ny=ny, do_vtk=do_vtk)

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
