# Townsend et al 2019 calculations of relaxation and eruption criteria

using JLD2, JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO

using CairoMakie, CSV, XLSX, DataFrames, LsqFit

data = DataFrame(XLSX.readtable("Onset_of_caldera_collapse_CSV.xlsx", "Systematics"))
# data = DataFrame(XLSX.readtable("Onset_of_caldera_collapse_CSV.xlsx", "Reference_run_variations"))

tectonic_setting = data[:, 2]
diameter_caldera = data[:, 3]
Topography = data[:, 4]
depth_chamber = data[:, 5]
radius_chamber = data[:, 6]
aspect_ratio = data[:, 7]
roof_ratio_chamber = data[:, 8]
friction_angle = data[:, 9]
friction_coeff = tand.(friction_angle)
eruption_time_s = data[:, 10]
eruption_volume = data[:, 11]
t_of_through_going_fault = data[:, 12]
underpressure_MPa = data[:, 16]
maximum_undepressure_MPa = data[:, 17]
Shear_band_orientation = data[:, 18]


Caldera_area_km2 = π .* (diameter_caldera ./ 2) .^2 ./ 1e6
radius_Caldera = diameter_caldera ./ 2
total_radius = radius_chamber .* aspect_ratio
density_host_rock = 2700.0 # kg/m^3
gravity = 9.81 # m/s^2
ΔPc = 20e6 # cirtical overpressure in Pa after Townsend et al 2019
thermal_diffusivity = 3.0/(2700.0 * 1050.0) # [W/mK] / [kg/m^3] * [J/kgK] = m^2/s
η_host_rock_Townsend2019 = 1e19 # Pa s
η_host_rock_JustCollapse_strong = 1e22 # Pa s
η_host_rock_JustCollapse_weak = 1e20 # Pa s
density_magma = 2400.0 # kg/m^3
M_in_normal = 3e-3 # [km3/yr]
M_in_increased = 1e-2 # [km3/yr]
# JustCollpase.jl ellipsoid chamber setup V = 4/3pi * a*b*a
V0 = @. 4/3 * π * (radius_chamber*1e3 * aspect_ratio)^2 * radius_chamber*1e3   # volume of the chamber in m^3
V0_km3 = V0 ./ 1e9


M_injection_normal_T2019 = (M_in_normal * 1e9 / 3.1536e7) * density_magma # kg/s
M_injection_increased_T2019 = (M_in_increased * 1e9 / 3.1536e7) * density_magma # kg/s
reference_model_V = @. 4/3 * π * (1750 * 2.5) * 1750 * (1750 * 2.5)

θ1_ref_normal = (3/4π) * (M_injection_normal_T2019 *inv(thermal_diffusivity*density_magma*(1750*2.5)))
θ2_ref_normal_strong = (3/4π) * (M_injection_normal_T2019 * η_host_rock_JustCollapse_strong) *inv(ΔPc * density_magma * (1750 * 2.5)^3)
θ2_ref_normal_weak = (3/4π) * (M_injection_normal_T2019 * η_host_rock_JustCollapse_weak) *inv(ΔPc * density_magma * (1750 * 2.5)^3)

θ1_ref_increased = (3/4π) * (M_injection_increased_T2019 * inv(thermal_diffusivity * density_magma * (1750 * 2.5)))
θ2_ref_increased_strong = (3/4π) * (M_injection_increased_T2019 * η_host_rock_JustCollapse_strong) *inv(ΔPc * density_magma * (1750 * 2.5)^3)
θ2_ref_increased_weak = (3/4π) * (M_injection_increased_T2019 * η_host_rock_JustCollapse_weak) *inv(ΔPc * density_magma * (1750 * 2.5)^3)

θ1_ref_townsend = (3/4π) * (M_injection_normal_T2019 *inv(thermal_diffusivity*density_magma*(1750*2.5)))
θ2_ref_townsend = (3/4π) * (M_injection_normal_T2019 * η_host_rock_Townsend2019) *inv(ΔPc * density_magma * (1750 * 2.5)^3)
θ1_ref_increased_townsend = (3/4π) * (M_injection_increased_T2019 * inv(thermal_diffusivity * density_magma * (1750 * 2.5)))
θ2_ref_increased_townsend = (3/4π) * (M_injection_increased_T2019 * η_host_rock_Townsend2019) *inv(ΔPc * density_magma * (1750 * 2.5)^3)

τ_in_T2019_normal = (density_magma * V0) *inv(M_injection_normal_T2019)  # [kg/m3] * [m3] / [kg/s] = [s]
τ_in_T2019_increased = (density_magma * V0) *inv(M_injection_increased_T2019)  # [kg/m3] * [m3] / [kg/s] = [s]

τ_relax_T2019 = η_host_rock_Townsend2019 / ΔPc
τ_relax_JustCollapse_strong = η_host_rock_JustCollapse_strong / ΔPc
τ_relax_JustCollapse_weak = η_host_rock_JustCollapse_weak / ΔPc

τ_cool = @. ((radius_chamber*1e3) * aspect_ratio)^2  * inv(thermal_diffusivity)

θ1_models_normal = @. (3/4π) * (M_injection_normal_T2019 * inv(thermal_diffusivity * density_magma * (total_radius*1e3)))
θ2_models_normal_strong = @. (3/4π) * (M_injection_normal_T2019 * η_host_rock_JustCollapse_strong) * inv(ΔPc * density_magma * (total_radius*1e3)^3)
θ2_models_normal_weak = @. (3/4π) * (M_injection_normal_T2019 * η_host_rock_JustCollapse_weak) * inv(ΔPc * density_magma * (total_radius*1e3)^3)
θ2_models_normal_townsend = @. (3/4π) * (M_injection_normal_T2019 * η_host_rock_Townsend2019) * inv(ΔPc * density_magma * (total_radius*1e3)^3)

θ1_models_increased = @. (3/4π) * (M_injection_increased_T2019 * inv(thermal_diffusivity * density_magma * (total_radius*1e3)))
θ2_models_increased_strong = @. (3/4π) * (M_injection_increased_T2019 * η_host_rock_JustCollapse_strong) * inv(ΔPc * density_magma * (total_radius*1e3)^3)
θ2_models_increased_weak = @. (3/4π) * (M_injection_increased_T2019 * η_host_rock_JustCollapse_weak) * inv(ΔPc * density_magma * (total_radius*1e3)^3)
θ2_models_increased_townsend = @. (3/4π) * (M_injection_increased_T2019 * η_host_rock_Townsend2019) * inv(ΔPc * density_magma * (total_radius*1e3)^3)


#scaling law degruyter & huber 2014
b1 = 2.4
b2 = 3.5
b3 = 2.5

θ1_vector = collect(LinRange(1e-3, 1e3, 200))

θ2_degruyter = @. (b3 * θ1_vector) * inv(b1*θ1_vector + b2 -1)

let

cmap = Makie.categorical_colors(:lipari10, 10)
colors = cmap[1:1:10]
# First lets plot the Townsend et al 2019 values with η_hostrock 1e19
fig = Figure(size = (1200, 800))

ax = Axis(fig[1, 1],
    xlabel = "θ₁ = τ_cool / τ_in",
    ylabel = "θ₂ = τ_relax / τ_in",
    xscale = log10,
    yscale = log10,
    xticklabelsize = 20,
    yticklabelsize = 20,
    xlabelsize = 24,
    ylabelsize = 24,
    # title = "Relaxation and Eruption Criteria"
)



#reference model with eta 1e19
scatter!(ax,
    θ1_ref_townsend,
    θ2_ref_townsend;
    marker = :star5,
    color = :grey,
    strokecolor = :grey,
    strokewidth = 2,
    label = "Reference model",
    markersize = 24
)

#reference model with eta 1e19 and increased injection rate
scatter!(ax,
    θ1_ref_increased_townsend,
    θ2_ref_increased_townsend;
    marker = :star5,
    color = :grey,
    strokecolor = :grey,
    strokewidth = 2,
    label = "Reference model",
    markersize = 24
)

# reference model with eta $(η_host_rock_JustCollapse)
scatter!(ax,
    θ1_ref_normal,
    θ2_ref_normal_strong;
    marker = :star5,
    # color = colors[1],
    color = :grey,
    label = "Reference model",
    strokecolor = :grey,
    strokewidth = 2,
    markersize = 24
)

scatter!(ax,
    θ1_ref_normal,
    θ2_ref_normal_weak;
    marker = :star5,
    # color = colors[4],
    color = :grey,
    label = "Reference model",
    strokecolor = :grey,
    strokewidth = 2,
    markersize = 24
)

# reference model with eta $(η_host_rock_JustCollapse) and increased injection rate
scatter!(ax,
    θ1_ref_increased,
    θ2_ref_increased_strong;
    marker = :star5,
    # color = colors[3],
    color = :grey,
    strokecolor = :grey,
    strokewidth = 2,
    label = "Reference model",
    markersize = 24
)

scatter!(ax,
    θ1_ref_increased,
    θ2_ref_increased_weak;
    marker = :star5,
    # color = colors[5],
    color = :grey,
    strokecolor = :grey,
    strokewidth = 2,
    label = "Reference model",
    markersize = 24
)

# Now the models w/ normal injection ra
scatter!(ax,
    θ1_models_normal,
    θ2_models_normal_strong;
    marker = :circle,
    color = colors[1],
    label = "Models with normal injection rate",
    markersize = 15
)

scatter!(ax,
    θ1_models_normal,
    θ2_models_normal_weak;
    marker = :circle,
    color = colors[4],
    label = "Models with normal injection rate",
    markersize = 15
)

# Now the models w/ increased injection rate
scatter!(ax,
    θ1_models_increased,
    θ2_models_increased_strong;
    marker = :diamond,
    color = colors[3],
    label = "Increased injection rate",
    markersize = 15
)

scatter!(ax,
    θ1_models_increased,
    θ2_models_increased_weak;
    marker = :diamond,
    color = colors[5],
    label = "Increased injection rate",
    markersize = 15
)

# Now the models w/ normal injection ra
scatter!(ax,
    θ1_models_normal,
    θ2_models_normal_strong;
    marker = :circle,
    color = colors[1],
    label = "Models with normal injection rate",
    markersize = 15
)
scatter!(ax,
    θ1_models_normal,
    θ2_models_normal_weak;
    marker = :circle,
    color = colors[4],
    label = "Models with normal injection rate",
    markersize = 15
)

# Now the models w/ increased injection rate
scatter!(ax,
    θ1_models_increased,
    θ2_models_increased_strong;
    marker = :diamond,
    color = colors[3],
    label = "Increased injection rate",
    markersize = 15
)

scatter!(ax,
    θ1_models_increased,
    θ2_models_increased_weak;
    marker = :diamond,
    color = colors[5],
    label = "Increased injection rate",
    markersize = 15
)

#now the models with the 1e19 Pa s viscosity from Townsend et al 2019
scatter!(ax,
    θ1_models_normal,
    θ2_models_normal_townsend;
    marker = :circle,
    color = (colors[6], 0.13),
    label = "Models with normal injection rate",
    markersize = 15
)

scatter!(ax,
    θ1_models_increased,
    θ2_models_increased_townsend;
    marker = :diamond,
    color = (colors[8], 0.13),
    alpha = 0.5,
    label = "Increased injection rate",
    markersize = 15
)

# xlims!(ax, 1e-3, 1e3)
xlims!(ax, 1e0, 1e3)
ylims!(ax, 1e-3, 1e4)
# hlines!(ax, [1.0], color=:black, linestyle=:dash)
# vlines!(ax, [1.0], color=:black, linestyle=:dashdot)

lines!(ax, θ1_vector, θ2_degruyter, color=:black, linestyle=:dash, label = "Degruyter & Huber 2014 - \n Scaling law for number of eruptions")
# fig[1,2] = Legend(fig, ax, "Townsend et al 2019 - Relaxation and Eruption Criteria", framevisible = true, merge=true, unique=true)
# fig[1,2] = Legend(fig, ax, framevisible = true, merge=true, unique=true)
axislegend(ax, framevisible = true, merge=true, unique=true, position = :rt, fontsize = 24, labelsize = 20, title = "Relaxation and Eruption Criteria", titlesize = 24)


# text!(ax, "Region 1: \n Eruption triggered by second boiling", position = (5e-3, 1e1), fontsize = 18)
# text!(ax, "Region 2: \n Eruption triggered \n by mass injection", position = (5e1, 2.5e0), fontsize = 18)
# text!(ax, "Region 3: \n No eruption", position = (5e1, 5e-2), fontsize = 18)
text!(ax, "Eruption triggered \n by mass injection", position = (1e2, 2.5e0), fontsize = 24)
text!(ax, "No eruption", position = (1e2, 5e-2), fontsize = 24)
display(fig)

save("./SmallScaleCaldera/Post_processing/Eruptibility_criteria.png", fig)
save("./SmallScaleCaldera/Post_processing/Eruptibility_criteria.svg", fig)
save("./SmallScaleCaldera/Post_processing/Eruptibility_criteria.pdf", fig)
# f_exsolution = @. inv(τ_cool * 1e-10 * 20e6)

# lines!(ax, f_exsolution, color=:orange, label = "Eruption triggered by second boiling")
# fig
end


# Number of eruptions (Degruyter & HUber 2014)

b1 = 2.4
b2 = 3.5
b3 = 2.5

η_host_rock = 1e20 # Pa s

fig = Figure(size = (1200, 800))
ax = Axis(fig[1, 1],
    xlabel = "Roof ratio of the chamber",
    ylabel = "Number of eruptions",
    title = "Number of eruptions - Degruyter & Huber 2014
")

for eta in [η_host_rock_Townsend2019, η_host_rock_JustCollapse_strong, η_host_rock_JustCollapse_weak]
    markers = [:circle, :diamond, :square]
# for i in 1:3
    cmap = Makie.categorical_colors(:lipari10, 10)
    colors = cmap[1:1:10]
    θ2_degruyter = (b3 * θ1_ref_townsend) * inv(b1*θ1_ref_townsend + b2 -1)

    A = @. (-4π * b3 * ΔPc * density_magma) * (total_radius*1e3).^3
    B = @. (4π * b2 * eta * thermal_diffusivity * density_magma) * (total_radius*1e3)
    C = @. 2*b3 * eta * M_injection_increased_T2019
    D = @. (4π * eta * thermal_diffusivity * density_magma) * (total_radius*1e3)

    N = @. (A + B + C) * inv(D)

    scatter!(ax, roof_ratio_chamber, N; label = "η_host_rock = $(eta)")#, marker = markers[i])
end
axislegend(ax, "Number of eruptions - Degruyter & Huber 2014", framevisible = true, merge=true, unique=true, position = :rb)
display(fig)
