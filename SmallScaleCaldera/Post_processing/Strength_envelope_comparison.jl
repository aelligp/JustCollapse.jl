## Comparison of different dislocation creep laws for strength envelopes

using GeoParams
using CairoMakie
using Interpolations
import GeoParams.Dislocation
import Interpolations: LinearInterpolation

using JLD2

# Custom function to handle temperature vectors directly
function StrengthEnvelopeWithTempVector(MatParam, Thickness, z_vec, T_vec, ε, nz=201)
    # This function computes strength envelope with a custom temperature profile

    # hardcoded input
    g = 9.81u"m/s^2"

    # nondimensionalize
    CharDim = GEO_units(length=10u"km", temperature=1000u"°C", stress=10u"MPa", viscosity=1.0e20u"Pa*s")
    MatParam = nondimensionalize(MatParam, CharDim)
    Thickness_nd = nondimensionalize(Thickness, CharDim)
    g_nd = nondimensionalize(g, CharDim)
    ε_nd = nondimensionalize(ε, CharDim)

    # Build depth grid
    Inter = cumsum(Thickness_nd)
    dz = Inter[end] / (nz - 1)
    z = collect(0:dz:Inter[end])

    # Interpolate temperature to new grid
    z_vec_nd = nondimensionalize(z_vec, CharDim)
    T_vec_nd = nondimensionalize(T_vec, CharDim)
    T = LinearInterpolation(z_vec_nd, T_vec_nd).(z)

    # Distribute phases
    nLayer = length(MatParam)
    Phases = ones(Int64, nz) * MatParam[1].Phase
    for i in 1:(nLayer-1)
        Phases[z .> Inter[i]] .= MatParam[i+1].Phase
    end

    # Pressure and density
    ρ = zeros(Float64, nz)
    P = GeoParams.LithPres(MatParam, Phases, ρ, T, dz, g_nd)

    # Solve for stress
    τ = zeros(Float64, nz)
    τ = GeoParams.solveStress(MatParam, Phases, ε_nd,  P, T)

    # Dimensionalize results
    z_dim = dimensionalize(z, u"km", CharDim)
    τ_dim = dimensionalize(τ, u"MPa", CharDim)
    T_dim = dimensionalize(T, u"°C", CharDim)

    return z_dim, τ_dim, T_dim
end

# Fixed parameters
C = 10.0u"MPa"  # Cohesion
ϕ = 30.0  # Friction angle
Thickness_total = 23.0u"km"
Tbot = 600.0u"°C"

# Option 1: Use linear temperature gradient (standard)
use_custom = false

if use_custom
    # JLD2.load("/Users/pascalaellig/Documents/PhD/JustCollapse.jl/SmallScaleCaldera/Post_processing/Creep_law_analysis/Temparature_Vector_Creep_Laws.jld2")
    Temp = (reverse(Temp_Vec_lithosphere) .* 1.0u"K" ) .|> u"°C"  # Convert from Kelvin to Celsius with units (reversed for surface-to-depth ordering)
    Temp = (reverse(Temp_Vec) .* 1.0u"K" ) .|> u"°C"  # Convert from Kelvin to Celsius with units (reversed for surface-to-depth ordering)
    z_vec = range(0.0u"km", Thickness_total, length=length(Temp))  # Depth vector
else
    Temp = LinTemp(0.0u"°C", Tbot)
end


@load "/Users/pascalaellig/Documents/PhD/JustCollapse.jl/SmallScaleCaldera/Post_processing/Creep_law_analysis/Temparature_Vector_Creep_Laws.jld2" Temp_Vec
Temp_custom = (reverse(Temp_Vec) .* 1.0u"K") .|> u"°C"
z_vec_custom = range(0.0u"km", Thickness_total, length=length(Temp_custom))

ε = 10^(-15) / u"s"  # Strain rate
# Define different rheologies to compare
creep_laws = [
    ("Granite (Carter 1987)ᵃ", Dislocation.granite_Carter_1987),
    ("Quartz Diorite (Hansen & Carter 1982)ᵇ", Dislocation.quartz_diorite_HansenCarter_1982),
    ("Diabase (Caristan 1982)ᶜ", Dislocation.diabase_Caristan_1982),
    ("Wet Quartzite (Ueda 2008)ᵈ", Dislocation.wet_quartzite_Ueda_2008),
    ("Plagioclase An75 (Ji 1993)ᵉ", Dislocation.plagioclase_An75_Ji_1993),
    ("Mafic Granulite (Wilks 1990)ᶠ", Dislocation.mafic_granulite_Wilks_1990),
    ("Strong Diabase (Mackwell 1998)ᵍ", Dislocation.strong_diabase_Mackwell_1998),
]

println("Computing strength envelopes for different creep laws...")
println("Parameters:")
println("  - Cohesion: $(C)")
println("  - Friction angle: $(ϕ)°")
println("  - Depth: $(Thickness_total)")
println("  - Temperature: 0°C to $(Tbot)")
println("  - Strain rate: $(ε)")

# Create figure
fig = Figure(size = (1400, 900), fontsize=14)

# Strength envelope plot
ax1 = Axis(fig[1, 1],
    title = "Strength Envelope Comparison",
    xlabel = "Maximum Strength [MPa]",
    ylabel = "Depth [km]",
    yreversed = true,
    xlabelsize = 24,
    ylabelsize = 24,
    xticklabelsize = 20,
    yticklabelsize = 20,
    titlesize = 30,

)

# Temperature profile plot
ax2 = Axis(fig[1, 2],
    title = "Temperature Profile",
    xlabel = "Temperature [°C]",
    ylabel = "Depth [km]",
    yreversed = true,
    xlabelsize = 24,
    ylabelsize = 24,
    xticklabelsize = 20,
    yticklabelsize = 20,
    titlesize = 30,
)

ax_inset_env = Axis(fig[1, 1],
    width=Relative(0.45),
    height=Relative(0.4),
    yreversed = true,
    halign=0.95,
    valign=0.925,
    xlabelsize = 18,
    ylabelsize = 18,
    xticklabelsize = 15,
    yticklabelsize = 15,
    backgroundcolor = (:white,1.0),
)

ax_inset_T = Axis(fig[1, 2],
    width=Relative(0.45),
    height=Relative(0.4),
    yreversed = true,
    halign=0.95,
    valign=0.925,
    xlabelsize = 18,
    ylabelsize = 18,
    xticklabelsize = 15,
    yticklabelsize = 15,
    backgroundcolor = (:white,1.0),
)

# Colors for different rheologies
colors = [:black, :blue, :green, :orange, :purple, :brown, :pink]
# colors = CairoMakie.categorical_colors(:lisbon10, length(creep_laws))
# Store results for legend
lines_created = []
labels_created = []

# Compute and plot each rheology
for (i, (name, creep_law)) in enumerate(creep_laws)
    println("Computing: $name")

    # Create material parameters
    mat = SetMaterialParams(;
        Name = name,
        Phase = 1,
        Density = PT_Density(ρ0 = 2.7e3kg/m^3, α = 3.0e-5/K, β = 1.0e-10/Pa),
        CreepLaws = SetDislocationCreep(creep_law),
        Plasticity = DruckerPrager(ϕ = ϕ, C = C),
    )

    rheology = (mat,)
    Thickness = [Thickness_total]

    try
        # Compute strength envelope
        if use_custom
            # Use custom temperature vector
            z, τ, T = StrengthEnvelopeWithTempVector(rheology, Thickness, z_vec, Temp, ε, 201)
        else
            # Use standard linear temperature
            z, τ, T = StrengthEnvelopeComp(rheology, Thickness, Temp, ε, 201)
        end

        # Convert to plain numbers for plotting
        z_plot = ustrip.(u"km", z)
        τ_plot = ustrip.(u"MPa", τ)
        T_plot = ustrip.(u"°C", T)

        # Plot strength envelope
        l = lines!(ax1, τ_plot, z_plot,
            linewidth=2.5,
            color=colors[i],
            label=name
        )

        push!(lines_created, l)
        push!(labels_created, name)

        # Plot temperature on first iteration only
        if i == 1
            lines!(ax2, T_plot, z_plot, linewidth=2, color=:black)
        end

        # Compute with custom temperature for insets
        z_ins, τ_ins, T_ins = StrengthEnvelopeWithTempVector(rheology, Thickness, z_vec_custom, Temp_custom, ε, 201)
        z_ins_plot = ustrip.(u"km", z_ins)
        τ_ins_plot = ustrip.(u"MPa", τ_ins)
        T_ins_plot = ustrip.(u"°C", T_ins)

        lines!(ax_inset_env, τ_ins_plot, z_ins_plot, linewidth=1.5, color=colors[i])
        if i == 1
            lines!(ax_inset_T, T_ins_plot, z_ins_plot, linewidth=2, color=:black)
        end

    catch e
        println("Error computing $name: $e")
    end
end

# Set axis limits
xlims!(ax1, low = 0)
xlims!(ax2, (0, ustrip(u"°C", Tbot) + 50))
xlims!(ax_inset_env, low = 0)
xlims!(ax_inset_T, (0, 1000))
ylims!(ax1, (ustrip(u"km", Thickness_total), 0))
ylims!(ax2, (ustrip(u"km", Thickness_total), 0))
ylims!(ax_inset_env, (ustrip(u"km", Thickness_total), 0))
ylims!(ax_inset_T, (ustrip(u"km", Thickness_total), 0))

# Add legend
Legend(fig[2, 1:2], ax1, framevisible=true, nbanks=2,
    # tellwidth=false, tellheight=true, labelsize=12, markerwidth=20, markerheight=10, padding=(10, 10, 10, 10))
    tellwidth=false, tellheight=true,orientation = :horizontal, fontsize = 24, "Dislocation Creep Laws", titleposition = :top, titlesize = 24, labelsize = 20)

println("\nPlot created successfully!")
println("Saving figure...")

# Save figure
# save("strength_envelope_comparison_magma.png", fig, px_per_unit=2)
# save("strength_envelope_comparison_magma.svg", fig, px_per_unit=2)
save("./SmallScaleCaldera/Post_processing/Strength_Envelope_plot.png", fig, px_per_unit=2)
save("./SmallScaleCaldera/Post_processing/Strength_Envelope_plot.svg", fig, px_per_unit=2)
save("./SmallScaleCaldera/Post_processing/Strength_Envelope_plot.pdf", fig)

# Display figure
display(fig)
