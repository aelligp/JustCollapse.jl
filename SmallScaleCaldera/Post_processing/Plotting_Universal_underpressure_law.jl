using CairoMakie, GeoParams, LaTeXStrings, CSV, XLSX, DataFrames, LsqFit

data = DataFrame(XLSX.readtable("Onset_of_caldera_collapse_CSV.xlsx", "Systematics"))

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
total_radius = radius_chamber .* aspect_ratio

Roof_ratio_radius_depth = @. depth_chamber / total_radius

Caldera_area_km2 = π .* (diameter_caldera ./ 2) .^2 ./ 1e6
radius_Caldera = diameter_caldera ./ 2
density_host_rock = 2700.0 # kg/m^3
gravity = 9.81 # m/s^2
C_rock = 10.0 # MPa, cohesion of the host rock
ρ = density_host_rock
g = gravity

tau_DP_eff(H_km, phi) = C_rock * cosd(phi) + (ρ * g * H_km * 1e3 / 2.0) / 1e6 * sind(phi)
tau_DP_eff_total(H_km, phi) = C_rock * cosd(phi) + (ρ * g * H_km * 1e3) / 1e6 * sind(phi)

Roof_ratio_lines = collect(LinRange(0.1, 3.0, length(total_radius)))

all_angles = [15.0, 20.0, 25.0, 30.0]

fig = Figure(size = (1200, 900))
ax = Axis(fig[1, 1];
    ylabel = "Underpressure ΔP [MPa]",
    xticks = [0.3, collect(0.5:0.5:2.5)...],
    xminorticks = 0.0:0.1:2.5,
    xticklabelsize = 18,
    yticklabelsize = 18,
    xlabelsize = 22,
    ylabelsize = 22,
)

ax2 = Axis(fig[2, 1];
    xlabel = "Roof ratio R = Depth/Width",
    ylabel = "Underpressure ΔP [MPa]",
    xticks = [0.3, collect(0.5:0.5:2.5)...],
    xminorticks = 0.0:0.1:2.5,
    xticklabelsize = 18,
    yticklabelsize = 18,
    xlabelsize = 22,
    ylabelsize = 22,
)
cmap = CairoMakie.categorical_colors(:lipari10, 10)
markers = [:circle, :rect, :diamond, :utriangle]

for (i, phi) in enumerate(all_angles)
    lower_threshold_tau = tau_DP_eff(5.665, phi) # Minimum τ_DP_eff at 1 km depth and lowest friction angle
    uppper_threshold_tau = tau_DP_eff(9.450, phi) # Maximum τ_DP_eff at 9.45 km depth and highest friction angle
    models = tau_DP_eff(5.8, phi)

    scaling_law_lower = (1.84 .* log.(Roof_ratio_lines) .+ 2.22) .* lower_threshold_tau
    scaling_law_upper = (1.84 .* log.(Roof_ratio_lines) .+ 2.22) .* uppper_threshold_tau
    scaling_law_models = (1.84 .* log.(Roof_ratio_lines) .+ 2.22) .* models
    scaling_law_bower =  2*log.(1 .+  2*Roof_ratio_lines) .* models
    scaling_law_cylinder = (2/√3 )* log.(1 .+ 2*Roof_ratio_lines) * models

    cline = cmap[i + 2]
    cband = cmap[i + 3]

    b1 = band!(ax, Roof_ratio_lines, scaling_law_lower, scaling_law_upper;
    color = cband, alpha = 0.2, strokecolor = cband, label = "H=5.6-9.45 km")

    ln1 =lines!(ax, Roof_ratio_lines, scaling_law_models;
        color = cline, linestyle = :solid, linewidth = 3,
        label = "Universal Fit - this study (H = 5.8 km)"
    )

    ln2 = lines!(ax2, Roof_ratio_lines, scaling_law_bower;
        color = (:black, 0.5), linestyle = :dot, linewidth = 2,
        label = "Analytical Solution: Hollow Sphere"
    )

    ln3 = lines!(ax2, Roof_ratio_lines, scaling_law_cylinder;
        color = (:black, 0.5), linestyle = :dashdot, linewidth = 2,
        label = "Analytical Solution: Cylinder"
    )


    mask = (friction_angle .== phi) #.& (tectonic_setting .== 0.0)
    s1 = scatter!(ax, roof_ratio_chamber[mask], abs.(underpressure_MPa[mask]); color = :black, alpha = 0.2, marker = markers[i], markersize = 8, label = "Models (ϕ=$(Int(phi))°)")

    mask1 = (friction_angle .== phi) .& (tectonic_setting .== 0.0)
    s2 = scatter!(ax2, roof_ratio_chamber[mask1], abs.(underpressure_MPa[mask1]); color = :black, alpha = 1.0, marker = markers[i], markersize = 8, label = "Models (ϕ=$(Int(phi))°)")

    (i == 4 && phi == 30.0) ? fig[3, 1] = Legend(fig, [b1, ln1, ln2, ln3, s2], ["Range for H=5.6-9.45 km", "Universal Fit", "Hollow Sphere (A = 2)", "Cylinder (A = 2/√3)", "Models"], tellheight = true, merge = true, unique = true, orientation = :horizontal, fontsize = 22, labelsize = 22) : nothing
end
scatter!(ax, [2.0], [(166 + 205)/2], color = cmap[1], markersize = 15, marker = :star5, label = "Katmai")
rangebars!(ax, [2.0], [166.0], [205.0], color = cmap[1])

# # Pinatubo: Roof ratio 2.4, Underpressure 265-312 MPa
scatter!(ax, [2.4], [(265 + 312)/2], color = cmap[3], markersize = 15, marker = :star5, label = "Pinatubo")
rangebars!(ax, [2.4], [265.0], [312.0], color = cmap[3])

ylims!(ax, -50, 325)
xlims!(ax, 0.0, 2.75)
xlims!(ax2, 0.0, 2.75)

text!(ax, 0.752, -5;  text = L"P_{under} = (1.84 \ln(R) + 3.2\ln(2)) \cdot [C \ \cos(\phi) + \bar{P} \ \sin(\phi)]", color = :black, fontsize = 24)

text!(ax2, 0.752, -5;  text = L"P_{under} = A \cdot \ln(1 + 2R) \cdot [C \ \cos(\phi) + \bar{P} \ \sin(\phi)]", color = :black, fontsize = 24)
display(fig)

save("./SmallScaleCaldera/Post_processing/Underpressure_scaling_universal_law_w_models_ylim_0_xlims_0_275.png", fig)
save("./SmallScaleCaldera/Post_processing/Underpressure_scaling_universal_law_w_models_ylim_0_xlims_0_275.pdf", fig)
save("./SmallScaleCaldera/Post_processing/Underpressure_scaling_universal_law_w_models_ylim_0_xlims_0_275.svg", fig)
