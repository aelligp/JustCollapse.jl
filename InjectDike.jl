using CUDA
# using Parameters
using Adapt
using ParallelStencil
# @init_parallel_stencil(CUDA, Float64, 2) #or (CUDA, Float64, 2) or (AMDGPU, Float64, 2)using Parameters
@init_parallel_stencil(Threads, Float64, 3) #or (CUDA, Float64, 2) or (AMDGPU, Float64, 2)using Parameters

using JustPIC
using JustPIC._2D
# using JustPIC._3D
using JustRelax
using Printf, Statistics, LinearAlgebra, GeoParams, CairoMakie, CellArrays
using StaticArrays
using ImplicitGlobalGrid#, MPI: MPI
const backend = CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
# const backend = CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

# model = PS_Setup(:CUDA, Float64, 2)            # initialize parallel stencil in 2D
model = PS_Setup(:Threads, Float64, 3)            # initialize parallel stencil in 2D
environment!(model)

## function from MTK
# @kwdef struct Dike    # stores info about dike
#     # Note: since we utilize the "Parameters.jl" package, we can add more keywords here w/out breaking the rest of the code
#     #
#     # We can also define only a few parameters here (like Q and ΔP) and compute Width/Thickness from that
#     # Or we can define thickness
#     Angle       ::  Array{Float64} =   [0.]                                  # Strike/Dip angle of dike
#     Type        ::  String          =   "ElasticDike"                          # Type of dike
#     T           ::  Float64         =   950.0                                 # Temperature of dike
#     E           ::  Float64         =   1.5e10                                # Youngs modulus (only required for elastic dikes)
#     ν           ::  Float64         =   0.3                                   # Poison ratio of host rocks
#     ΔP          ::  Float64         =   1e6;                                  # Overpressure of elastic dike
#     Q           ::  Float64         =   1000;                                 # Volume of elastic dike
#     W           ::  Float64         =   (3*E*Q/(16*(1-ν^2)*ΔP))^(1.0/3.0);    # Width of dike/sill
#     H           ::  Float64         =   8*(1-ν^2)*ΔP*W/(π*E);                 # (maximum) Thickness of dike/sill
#     Center      ::  Array{Float64}  =   [20e3 ; -10e3]                        # Center
#     Phase       ::  Int64           =   2;                                    # Phase of newly injected magma
# end

@kwdef struct Dike{_T,N}
    Angle       ::  NTuple{N, _T} =   ntuple(i -> 0.0, N)
    Type        ::  Symbol =   :ElasticDike
    T           ::  _T =   950.0
    E           ::  _T =   1.5e10
    ν           ::  _T =   0.3
    ΔP          ::  _T =   1e6
    Q           ::  _T =   1000
    W           ::  _T =   (3*E*Q/(16*(1-ν^2)*ΔP))^(1.0/3.0)
    H           ::  _T =   8*(1-ν^2)*ΔP*W/(π*E)
    Center      ::  NTuple{N, _T} =   ntuple(i -> 20e3, N)
    Phase       ::  Int64 =   2
end
Adapt.@adapt_structure Dike

## function from MTK
function DisplacementAroundPennyShapedDike(dike::Dike, CartesianPoint::SVector, dim)

    # extract required info from dike structure
    (;ν,E,W, H) = dike;

    Displacement = Vector{Float64}(undef, dim);

    # Compute r & z; note that the Sun solution is defined for z>=0 (vertical)
    if      dim==2; r = sqrt(CartesianPoint[1]^2);                       z = abs(CartesianPoint[2]);
    elseif  dim==3; r = sqrt(CartesianPoint[1]^2 + CartesianPoint[2]^2); z = abs(CartesianPoint[3]); end

    if r==0; r=1e-3; end

    B::Float64   =  H;                          # maximum thickness of dike
    a::Float64   =  W/2.0;                      # radius

    # note, we can either specify B and a, and compute pressure p and injected volume Q
    # Alternatively, it is also possible to:
    #       - specify p and a, and compute B and Q
    #       - specify volume Q & p and compute radius and B
    #
    # What is best to do is to be decided later (and doesn't change the code below)
    Q   =   B*(2pi*a.^2)/3.0;               # volume of dike (follows from eq. 9 and 10a)
    p   =   3E*Q/(16.0*(1.0 - ν^2)*a^3);    # overpressure of dike (from eq. 10a) = 3E*pi*B/(8*(1-ν^2)*a)

    # Compute displacement, using complex functions
    R1  =   sqrt(r^2. + (z - im*a)^2);
    R2  =   sqrt(r^2. + (z + im*a)^2);

    # equation 7a:
    U   =   im*p*(1+ν)*(1-2ν)/(2pi*E)*( r*log( (R2+z+im*a)/(R1 +z- im*a))
                                                - r/2*((im*a-3z-R2)/(R2+z+im*a)
                                                + (R1+3z+im*a)/(R1+z-im*a))
                                                - (2z^2 * r)/(1 -2ν)*(1/(R2*(R2+z+im*a)) -1/(R1*(R1+z-im*a)))
                                                + (2*z*r)/(1-2ν)*(1/R2 - 1/R1) );
    # equation 7b:
    W   =       2*im*p*(1-ν^2)/(pi*E)*( z*log( (R2+z+im*a)/(R1+z-im*a))
                                                - (R2-R1)
                                                - 1/(2*(1-ν))*( z*log( (R2+z+im*a)/(R1+z-im*a)) - im*a*z*(1/R2 + 1/R1)) );

    # Displacements are the real parts of U and W.
    #  Note that this is the total required elastic displacement (in m) to open the dike.
    #  If we only want to open the dike partially, we will need to normalize these values accordingly (done externally)
    Uz   =  real(W);  # vertical displacement should be corrected for z<0
    Ur   =  real(U);
    if (CartesianPoint[end]<0); Uz = -Uz; end
    if (CartesianPoint[1]  <0); Ur = -Ur; end

    if      dim==2
        Displacement = [Ur;Uz]
    elseif  dim==3
        # Ur denotes the radial displacement; in 3D we have to decompose this in x and y components
        x   = abs(CartesianPoint[1]); y = abs(CartesianPoint[2]);
        Ux  = x/r*Ur; Uy = y/r*Ur;

        Displacement = [Ux;Uy;Uz]
    end

    return Displacement, B, p
end

## for plotting the particles
function plot_particles(particles, pPhases)
    p = particles.coords
    # pp = [argmax(p) for p in phase_ratios.center] #if you want to plot it in a heatmap rather than scatter
    ppx, ppy = p
    pxv = ppx.data[:]
    pyv = ppy.data[:]
    clr = pPhases.data[:]
    # clrT = pT.data[:]
    idxv = particles.index.data[:]
    f,ax,h=scatter(Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), colormap=:roma)
    Colorbar(f[1,2], h)
    f
end


function Inject_dike_particles2D!(phases, pT, particles::Particles, dike::Dike)
    (; coords, index) = particles
    px, py = coords
    ni     = size(px)
    @parallel_indices (I...) function Inject_dike_particles!(phases, pT, px, py, index, dike)
        for ip in JustRelax.cellaxes(px)
            !(JustRelax.@cell(index[ip, I...])) && continue
            px_ip = JustRelax.@cell   px[ip, I...]
            py_ip = JustRelax.@cell   py[ip, I...]

            (;Angle, Center, H, W, Phase, T) = dike
            α = Angle[1]
            Δ = H

            RotMat = SA[cos(α) -sin(α); sin(α) cos(α)]
            px_rot = px_ip.-Center[1]
            py_rot = py_ip.-Center[2]

            pt_rot = RotMat*SA[px_rot,py_rot]
            px_rot = pt_rot[1]
            py_rot = pt_rot[2]

            # # not ideal as it plots the phase and then actually displaces everything
            in, on = isinside_dike(pt_rot, W, H);
            # Define displacement before the if blocks
            displacement = [0.0, 0.0]
            if on
                JustRelax.@cell phases[ip, I...] = Float64(Phase) # magma
                JustRelax.@cell pT[ip, I...] = T
                displacement, Bmax, overpressure = DisplacementAroundPennyShapedDike(dike, SA[px_rot,  py_rot],2)
                displacement .= displacement*H
            end
            if in
                JustRelax.@cell phases[ip, I...] = Float64(Phase) # magma
                JustRelax.@cell pT[ip, I...] = T
                displacement, Bmax, overpressure = DisplacementAroundPennyShapedDike(dike, SA[px_rot,  py_rot],2)
                displacement .= displacement*H
            end

            px_ip = px_rot + displacement[1]
            py_ip = py_rot + displacement[2]


            # RotatePoint_2D_new!(px,py, SA[px_rot,py_rot], RotMat')
            pt_rot = RotMat'*SA[px_ip,py_ip]
            px_ip = pt_rot[1] .+ Center[1]
            py_ip = pt_rot[2] .+ Center[2]

            JustRelax.@cell px[ip, I...] = px_ip
            JustRelax.@cell py[ip, I...] = py_ip
        end

        return nothing
    end
    @parallel (@idx ni) Inject_dike_particles!(phases, pT, coords..., index, dike)
end



function Inject_dike_particles3D!(phases, pT, particles::Particles, dike::Dike)
    (; coords, index) = particles
    px, py = coords
    ni     = size(px)
    @parallel_indices (I...) function Inject_dike_particles!(phases, pT, px, py, pz, index, dike)
        for ip in JustRelax.cellaxes(px)
            !(JustRelax.@cell(index[ip, I...])) && continue
            px_ip = JustRelax.@cell   px[ip, I...]
            py_ip = JustRelax.@cell   py[ip, I...]
            pz_ip = JustRelax.@cell   pz[ip, I...]

            (;Angle, Center, H, W, Phase, T) = dike
            Δ = H
            α,β             =   Angle[1], Angle[end];
            RotMat_y        =   SA[cosd(α) 0.0 -sind(α); 0.0 1.0 0.0; sind(α) 0.0 cosd(α)  ];                      # perpendicular to y axis
            RotMat_z        =   SA[cosd(β) -sind(β) 0.0; sind(β) cosd(β) 0.0; 0.0 0.0 1.0  ];                      # perpendicular to z axis
            RotMat          =   RotMat_y*RotMat_z;

            # RotMat = SA[cos(α) -sin(α); sin(α) cos(α)]
            px_rot = px_ip .- Center[1]
            py_rot = py_ip .- Center[2]
            pz_rot = pz_ip .- Center[3]

            pt_rot = RotMat*SA[px_rot,py_rot, pz_rot]
            px_rot = pt_rot[1]
            py_rot = pt_rot[2]
            pz_rot = pt_rot[3]

            # # not ideal as it plots the phase and then actually displaces everything
            in, on = isinside_dike(pt_rot, W, H);
            # Define displacement before the if blocks
            displacement = [0.0, 0.0, 0.0]
            if on
                JustRelax.@cell phases[ip, I...] = Float64(Phase) # magma
                JustRelax.@cell pT[ip, I...] = T
                displacement, Bmax, overpressure = DisplacementAroundPennyShapedDike(dike, SA[px_rot,  py_rot],2)
                displacement .= displacement*Bmax   # not necessary anymore
            end
            if in
                JustRelax.@cell phases[ip, I...] = Float64(Phase) # magma
                JustRelax.@cell pT[ip, I...] = T
            end

            px_ip = px_rot + displacement[1]
            py_ip = py_rot + displacement[2]
            pz_ip = pz_rot + displacement[3]

            # RotatePoint_2D_new!(px,py, SA[px_rot,py_rot], RotMat')
            pt_rot = RotMat'*SA[px_ip,py_ip, pz_ip]
            px_ip = pt_rot[1] .+ Center[1]
            py_ip = pt_rot[2] .+ Center[2]
            pz_ip = pt_rot[3] .+ Center[3]

            JustRelax.@cell px[ip, I...] = px_ip
            JustRelax.@cell py[ip, I...] = py_ip
            JustRelax.@cell pz[ip, I...] = pz_ip

        end

        return nothing
    end
    @parallel (@idx ni) Inject_dike_particles!(phases, pT, coords..., index, dike)

end

function isinside_dike(pt, W, H)
    dim =   length(pt)
    in  =   false;
    on  =   false;
    eq_ellipse = 100.0;
    tolerance = 0.9e-1;  # Define a tolerance for the comparison

    if dim==2
      eq_ellipse = (pt[1]^2.0)/((W/2.0)^2.0) + (pt[2]^2.0)/((H/2.0)^2.0); # ellipse
    elseif dim==3
        # radius = sqrt(*)x^2+y^2)
        eq_ellipse = (pt[1]^2.0 + pt[2]^2.0)/((W/2.0)^2.0) + (pt[3]^2.0)/((H/2.0)^2.0); # ellipsoid
    else
        error("Unknown # of dimensions: $dim")
    end

    if eq_ellipse <= 1.0
        in = true;
    end
    if abs(eq_ellipse - 1.0) < tolerance  # Check if eq_ellipse is close to 1.0 within the tolerance
        on = true;
    end
    return in, on
end

###################################
#MWE
###################################
#-------JustRelax parameters-------------------------------------------------------------

ar = 1 # aspect ratio
n = 96
nx = n * ar - 2
ny = n - 2
nz = n - 2
igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny,1; init_MPI=true)...)
else
    igg
end

# Domain setup for JustRelax
lx, ly, lz          = 10e3, 10e3, 10e3 # nondim if CharDim=CharDim
li              = lx, ly, lz
ni              = nx,nz
ni              = nx, ny, nz
b_width         = (4, 4, 1) #boundary width
origin          = 0.0, 0.0, -lz
igg             = igg
di              = @. li / ni # grid step in x- and y-direction
# di = @. li / (nx_g(), ny_g()) # grid step in x- and y-direction
grid            = Geometry(ni, li; origin = origin)
(; xci, xvi)    = grid # nodes at the center and vertices of the cells
#----------------------------------------------------------------------
# Initialize particles -------------------------------
nxcell, max_xcell, min_xcell = 20, 40, 15
particles = init_particles(
    backend, nxcell, max_xcell, min_xcell, xvi..., di..., ni...
)
# temperature
pT, pPhases      = init_cell_arrays(particles, Val(3))
particle_args = (pT, pPhases)
# init_dike_phase(dike_Phase,particles)
W_in, H_in              = 1e3,0.5e3; # Width and thickness of dike
T_in                    = 1000
H_ran, W_ran            = (xvi[1].stop, xvi[2].start).* [0.2;0.3];                          # Size of domain in which we randomly place dikes and range of angles
Dike_Type               = "ElasticDike"
# Tracers                 = StructArray{Tracer}(undef, 1);
nTr_dike                = 300;
# cen              = ((xvi[1].stop, xvi[2].stop).+(xvi[1].start,xvi[2].start))./2.0 .+ rand(-0.5:1e-3:0.5, 2).*[W_ran;H_ran];
cen3D              = ((xvi[1].stop, xvi[2].stop,xvi[3].stop).+(xvi[1].start,xvi[2].start,xvi[3].start))./2.0 .+ rand(-0.5:1e-3:0.5, 3).*[W_ran;W_ran;H_ran];
T_buffer    = @zeros(ni .+ 1)
Told_buffer = similar(T_buffer)
# if cen[end] < -5e3
#     Angle_rand = [rand(80.0:0.1:100.0)] # Orientation: near-vertical @ depth
# else
#     Angle_rand = [rand(-10.0:0.1:10.0)] # Orientation: near-vertical @ shallower depth
# end
if cen3D[end] < -5e3
    Angle_rand = [rand(80.0:0.1:100.0); rand(0:360)] # Orientation: near-vertical @ depth
else
    Angle_rand = [rand(-10.0:0.1:10.0); rand(0:360)] # Orientation: near-vertical @ shallower depth
end

dike      =   Dike(Angle=Angle_rand, W=W_in, H=H_in, Type=Dike_Type, T=T_in, Center=cen3D, #=ΔP=10e6,=#Phase =2 );
# dike      =   Dike(Angle=Angle_rand, W=W_in, H=H_in, Type=Dike_Type, T=T_in, Center=cen, #=ΔP=10e6,=#Phase =2 );

Inject_dike_particles2D!(pPhases,pT,particles, dike)
# Inject_dike_particles3D!(pPhases,pT,particles, dike)
plot_particles(particles, pPhases)
# Advection --------------------
# advect particles in space
advection_RK!(particles, @velocity(stokes), grid_vx, grid_vy, dt, 2 / 3)
# advect particles in memory
move_particles!(particles, xvi, particle_args)
grid2particle_flip!(pT, xvi, T_buffer, Told_buffer, particles)
# this will be a problem - no particles have escaped and therefore it does not inject
# if we force inect it will probably destroy the code
inject = check_injection(particles)
inject && inject_particles_phase!(particles, pPhases, (pT, ), (T_buffer,), xvi)
plot_particles(particles, pPhases)


function init_dike_phase!(phases, particles)
    ni = size(phases)

    @parallel_indices (I...) function init_dike_phase(phases, px, py, index)
    @inbounds for ip in JustRelax.JustRelax.cellaxes(phases)
        # quick escape
        # JustRelax.@cell(index[ip, i, j]) == 0 && continue

        x = JustRelax.@cell px[ip, i, j]
        y = -(JustRelax.@cell py[ip, i, j])
        JustRelax.@cell phases[ip, i, j] = 2.0 # magma
        end
        return nothing
    end

    @parallel (@idx ni) init_dike_phase(phases, particles.coords..., particles.index)
end




"""
    T, Tracers, dike_poly = AddDike(T,Tracers, Grid, dike,nTr_dike)

Adds a dike, described by the dike polygon dike_poly, to the temperature field T (defined at points Grid).
Also adds nTr_dike new tracers randomly distributed within the dike, to the
tracers array Tracers.

"""
function AddDike(Tfield, Grid,dike)

    dim         =   length(Grid);
    (;Angle,Center,W,H,T, Phase) = dike;
    PhaseDike = Phase;

    if dim==2
        α           =    Angle[1];
        RotMat      =    SA[cosd(α) -sind(α); sind(α) cosd(α)];
    elseif dim==3
        α,β             =   Angle[1], Angle[end];
        RotMat_y        =   SA[cosd(α) 0.0  -sind(α); 0.0 1.0 0.0; sind(α) 0.0 cosd(α)  ];                      # perpendicular to y axis
        RotMat_z        =   SA[cosd(β) -sind(β) 0.0; sind(β) cosd(β) 0.0; 0.0 0.0 1.0   ];                      # perpendicular to z axis
        RotMat          =   RotMat_y*RotMat_z;
    end

    # Add dike to temperature field
    if dim==2
        x,z = Grid[1], Grid[2];
        for i in eachindex(x)
            for j in eachindex(z)
                pt      =  SA[x[i];z[j]] - Center;
                pt_rot  =   RotMat*pt;                      # rotate
                in      =   isinside_dike(pt_rot, dike);
                if in
                    Tfield[i,j] = T;
                end
            end
        end

    elseif dim==3
        x,y,z = Grid[1], Grid[2], Grid[3]
        for ix=1:length(x)
            for iy=1:length(y)
                for iz=1:length(z)
                    pt      = SA[(x[ix], y[iy], z[iz])] - Center;
                    pt_rot  = RotMat*pt;          # rotate and shift
                    in      = isinside_dike(pt_rot, dike);
                    if in
                        Tfield[ix,iy,iz] = T;
                    end
                end
            end
        end

    end

    return Tfield#, Tr;

end


T_buffer    = @zeros(ni .+ 1)
Told_buffer = similar(T_buffer)
# Tracers, Tnew_cpu, Vol, dike_poly, Velo  =   InjectDike(Tracers, Tnew_cpu, xvi, dike, nTr_dike);   # Add dike, move hostrocks )
AddDike(T_buffer, xvi, dike)


dp = 2*pi/nump
p = 0.0:.01:2*pi;
a_ellipse = dike.W/2.0;
b_ellipse = dike.H/2.0;
x =  cos.(p)*a_ellipse
z = -sin.(p)*b_ellipse .+ dike.Center[2];
poly = [x,z];
