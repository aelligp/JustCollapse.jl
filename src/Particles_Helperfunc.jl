## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------
using JustRelax
import JustRelax.@cell
using Adapt

function copyinn_x!(A, B)

    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end

    @parallel f_x(A, B)
end


@kwdef struct Dike{_T,N}
    Angle       ::  NTuple{N, _T} =   ntuple(i -> 0.0, N)
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
                # displacement, Bmax, overpressure = DisplacementAroundPennyShapedDike(dike, SA[px_rot,  py_rot],2)
                # displacement .= displacement*H
            end
            if in
                JustRelax.@cell phases[ip, I...] = Float64(Phase) # magma
                JustRelax.@cell pT[ip, I...] = T
                # displacement, Bmax, overpressure = DisplacementAroundPennyShapedDike(dike, SA[px_rot,  py_rot],2)
                # displacement .= displacement*H
            end

            px_ip = px_rot #+ displacement[1]
            py_ip = py_rot #+ displacement[2]


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
