using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO

using CairoMakie, XLSX, DataFrames

data = DataFrame(XLSX.readtable("Onset_of_caldera_collapse.xlsx", "Systematics"))

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
thermal_diffusivity = 3.0/(density_host_rock * 1050.0) # [W/mK] / [kg/m^3] * [J/kgK] = m^2/s
η_host_rock_Townsend2019 = 1e19 # Pa s
η_host_rock_JustCollapse_strong = 1e22 # Pa s
η_host_rock_JustCollapse_weak = 1e20 # Pa s
density_magma = 2400.0 # kg/m^3
M_in_normal = 3e-3 # [km3/yr]
M_in_increased = 1e-2 # [km3/yr]
# JustCollpase.jl ellipsoid chamber setup V = 4/3pi * a*b*a
V0 = @. (radius_chamber*1e3 * aspect_ratio)^2 * radius_chamber*1e3   # volume of the chamber in m^3
V0_km3 = V0 ./ 1e9
radius_ref_chamber = @. (1750 * 2.5)^2 * 1750 # volume of the reference chamber in km^3
radius_chamber = V0

M_injection_normal_T2019 = (M_in_normal * 1e9 / 3.1536e7) * density_magma # kg/s
M_injection_increased_T2019 = (M_in_increased * 1e9 / 3.1536e7) * density_magma # kg/s
reference_model_V = @. 4/3 * π * (1750 * 2.5) * 1750 * (1750 * 2.5)

θ1_ref_normal = (3/4π) * (M_injection_normal_T2019 *inv(thermal_diffusivity*density_magma*(1750*2.5)))
θ2_ref_normal_strong = (3/4π) * (M_injection_normal_T2019 * η_host_rock_JustCollapse_strong) *inv(ΔPc * density_magma * radius_ref_chamber)
θ2_ref_normal_weak = (3/4π) * (M_injection_normal_T2019 * η_host_rock_JustCollapse_weak) *inv(ΔPc * density_magma * radius_ref_chamber)

θ1_ref_increased = (3/4π) * (M_injection_increased_T2019 * inv(thermal_diffusivity * density_magma * (1750 * 2.5)))
θ2_ref_increased_strong = (3/4π) * (M_injection_increased_T2019 * η_host_rock_JustCollapse_strong) *inv(ΔPc * density_magma * radius_ref_chamber)
θ2_ref_increased_weak = (3/4π) * (M_injection_increased_T2019 * η_host_rock_JustCollapse_weak) *inv(ΔPc * density_magma * radius_ref_chamber)

θ1_ref_townsend = (3/4π) * (M_injection_normal_T2019 *inv(thermal_diffusivity*density_magma*(1750*2.5)))
θ2_ref_townsend = (3/4π) * (M_injection_normal_T2019 * η_host_rock_Townsend2019) *inv(ΔPc * density_magma * radius_ref_chamber)
θ1_ref_increased_townsend = (3/4π) * (M_injection_increased_T2019 * inv(thermal_diffusivity * density_magma * (1750 * 2.5)))
θ2_ref_increased_townsend = (3/4π) * (M_injection_increased_T2019 * η_host_rock_Townsend2019) *inv(ΔPc * density_magma * radius_ref_chamber)

τ_in_T2019_normal = (density_magma * V0) *inv(M_injection_normal_T2019)  # [kg/m3] * [m3] / [kg/s] = [s]
τ_in_T2019_increased = (density_magma * V0) *inv(M_injection_increased_T2019)  # [kg/m3] * [m3] / [kg/s] = [s]

τ_relax_T2019 = η_host_rock_Townsend2019 / ΔPc
τ_relax_JustCollapse_strong = η_host_rock_JustCollapse_strong / ΔPc
τ_relax_JustCollapse_weak = η_host_rock_JustCollapse_weak / ΔPc

τ_cool = @. ((radius_chamber*1e3) * aspect_ratio)^2  * inv(thermal_diffusivity)

θ1_models_normal = @. (3/4π) * (M_injection_normal_T2019 * inv(thermal_diffusivity * density_magma * (total_radius*1e3)))
θ2_models_normal_strong = @. (3/4π) * (M_injection_normal_T2019 * η_host_rock_JustCollapse_strong) * inv(ΔPc * density_magma * radius_chamber)
θ2_models_normal_weak = @. (3/4π) * (M_injection_normal_T2019 * η_host_rock_JustCollapse_weak) * inv(ΔPc * density_magma * radius_chamber)
θ2_models_normal_townsend = @. (3/4π) * (M_injection_normal_T2019 * η_host_rock_Townsend2019) * inv(ΔPc * density_magma * radius_chamber)

θ1_models_increased = @. (3/4π) * (M_injection_increased_T2019 * inv(thermal_diffusivity * density_magma * (total_radius*1e3)))
θ2_models_increased_strong = @. (3/4π) * (M_injection_increased_T2019 * η_host_rock_JustCollapse_strong) * inv(ΔPc * density_magma * radius_chamber)
θ2_models_increased_weak = @. (3/4π) * (M_injection_increased_T2019 * η_host_rock_JustCollapse_weak) * inv(ΔPc * density_magma * radius_chamber)
θ2_models_increased_townsend = @. (3/4π) * (M_injection_increased_T2019 * η_host_rock_Townsend2019) * inv(ΔPc * density_magma * radius_chamber)


#scaling law degruyter & huber 2014
b1 = 2.4
b2 = 3.5
b3 = 2.5

θ1_vector = collect(LinRange(1e-3, 1e3, 400))

θ2_degruyter = @. (b3 * θ1_vector) * inv(b1*θ1_vector + b2 -1)
N_eruptions = @. b1 * θ1_vector + b2 - b3 * (θ1_vector * inv(θ2_degruyter))


let

cmap = Makie.categorical_colors(:lipari10, 10)
cmap_bg = Makie.categorical_colors(:vik10, 10)
colors = cmap[1:1:10]
colors_bg = cmap_bg[1:1:10]
# First lets plot the Townsend et al 2019 values with η_hostrock 1e19
fig = Figure(size = (1200, 800))

ax = Axis(fig[1, 1],
    xlabel = L"θ_1 = τ_\textrm{cool} / τ_\textrm{in}",
    ylabel = L"θ_2 = τ_\textrm{relax} / τ_\textrm{in}",
    xscale = log10,
    yscale = log10,
    xticklabelsize = 20,
    yticklabelsize = 20,
    xlabelsize = 30,
    ylabelsize = 30,
    xminorticksvisible=true,
    xminorticks=IntervalsBetween(9),
    yminorticksvisible=true,
    yminorticks=IntervalsBetween(9),
)

idx_split = findfirst(x -> x >= 1.0, θ1_vector)

second_boiling_pts = [
    Point2f(1e-3, 1e-3),
    Point2f(1.0,  0.32),
    Point2f(1.0,  1e5),
    Point2f(1e-3, 1e5),
]
poly!(ax, second_boiling_pts; color = (colors_bg[2], 0.25))

mass_inj_pts = vcat(
    [Point2f(1.0, 0.32)],
    [Point2f(x, y) for (x, y) in zip(θ1_vector[idx_split:end], θ2_degruyter[idx_split:end])],
    [Point2f(1e3, 1e5)],
    [Point2f(1.0, 1e5)],
)
poly!(ax, mass_inj_pts; color = (colors_bg[8], 0.25))

no_erupt_pts = vcat(
    [Point2f(1e3, θ2_degruyter[1])],
    [Point2f(x, y) for (x, y) in zip(θ1_vector, θ2_degruyter)],
)
poly!(ax, no_erupt_pts; color = (:grey, 0.25))

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

# xlims!(ax, 1e-3, 1e3)
xlims!(ax, 1e-3, 1e3)
ylims!(ax, 1e-3, 1e5)
# hlines!(ax, [1.0], color=:black, linestyle=:dash)
vlines!(ax, [1.0]; ymin = 0.32, color=:black, linestyle=:dashdot)

lines!(ax, θ1_vector, θ2_degruyter, color=:black, linestyle=:dash, label = "Degruyter & Huber 2014 - \n Scaling law for number of eruptions")
# fig[1,2] = Legend(fig, ax, "Townsend et al 2019 - Relaxation and Eruption Criteria", framevisible = true, merge=true, unique=true)
# fig[1,2] = Legend(fig, ax, framevisible = true, merge=true, unique=true)
ref_model = MarkerElement(color = :grey, marker = :star5, markersize = 18)
models_normal = MarkerElement(color = colors[1], marker = :circle, markersize = 18)
models_normal = MarkerElement(color = colors[1], marker = :circle, markersize = 18)
models_increased = MarkerElement(color = colors[3], marker = :diamond, markersize = 18)
models_1e22 = MarkerElement(color = colors[3], marker = :circle, markersize = 18)
models_1e20 = MarkerElement(color = colors[5], marker = :circle, markersize = 18)
models_1e19 = MarkerElement(color = colors[6], marker = :circle, markersize = 18)
scaling_degruyter = LineElement(color = :black, linestyle = :dash, linewidth = 2)
axislegend(ax, [ref_model, models_normal, models_increased, models_1e22, models_1e20, models_1e19, scaling_degruyter],
    ["Reference model", "Normal injection rate", "Increased injection rate", L"\eta_r =  10^{22} ", L"\eta_r =  10^{20} ", L"\eta_r =  10^{19} ", "Degruyter & Huber 2014 -\nRegime Boundaries"],
    framevisible = true, merge=true, unique=true, position = :lt, fontsize = 22, labelsize = 22
)

text!(ax, "Eruption triggered \nby mass injection", position = (5e1, 2e2), fontsize = 26)
text!(ax, "No eruption", position = (5e1, 5e-2), fontsize = 26)
text!(ax, "Eruption triggered \nby second boiling", position = (5e-3, 2.5e0), fontsize = 26)
display(fig)

save("./SmallScaleCaldera/Post_processing/Eruptibility_criteria.png", fig)
save("./SmallScaleCaldera/Post_processing/Eruptibility_criteria.svg", fig)
save("./SmallScaleCaldera/Post_processing/Eruptibility_criteria.pdf", fig)
end
