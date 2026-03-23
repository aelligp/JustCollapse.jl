using CairoMakie, GeoParams, LaTeXStrings, CSV, XLSX, DataFrames

# data = DataFrame(XLSX.readtable("Onset_of_caldera_collapse_CSV.xlsx", "Systematics"))
data = DataFrame(XLSX.readtable("Onset_of_caldera_collapse_CSV.xlsx", "All_models"))

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

Roof_ratio_lines = collect(LinRange(0.1, 3.25, length(total_radius)))

all_angles = [15.0, 20.0, 25.0, 30.0]

analytical_solution = false

let
analytical_solution == false ? fig = Figure(size = (1200, 600)) : fig = Figure(size = (1200, 900))
xlab = (analytical_solution == false ? "Roof ratio of chamber (depth / width)" : nothing)
ax = Axis(fig[1, 1];
    xlabel = analytical_solution == false ? "Roof ratio of chamber (depth / width)" : "",
    ylabel = "Underpressure [MPa]",
    xticks = [0.3, collect(0.5:0.5:2.5)...],
    xminorticks = 0.0:0.1:2.5,
    xticklabelsize = 18,
    yticklabelsize = 18,
    xlabelsize = 22,
    ylabelsize = 22,
)

analytical_solution == true ? (ax2 = Axis(fig[2, 1];
    xlabel = "Roof ratio R = Depth/Width",
    ylabel = "Underpressure [MPa]",
    xticks = [0.3, collect(0.5:0.5:2.5)...],
    xminorticks = 0.0:0.1:2.5,
    xticklabelsize = 18,
    yticklabelsize = 18,
    xlabelsize = 22,
    ylabelsize = 22,
)) : nothing
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
    if analytical_solution == true
    ln2 = lines!(ax2, Roof_ratio_lines, scaling_law_bower;
        color = (:black, 0.5), linestyle = :dot, linewidth = 2,
        label = "Analytical Solution: Hollow Sphere"
    )

    ln3 = lines!(ax2, Roof_ratio_lines, scaling_law_cylinder;
        color = (:black, 0.5), linestyle = :dashdot, linewidth = 2,
        label = "Analytical Solution: Cylinder"
    )
    end

    mask = (friction_angle[1:252] .== phi) #.& (tectonic_setting .== 0.0)
    s1 = scatter!(ax, roof_ratio_chamber[1:252][mask], -(underpressure_MPa[1:252][mask]); color = :black, alpha = 0.3, marker = markers[i], markersize = 10, label = "Models")
    s5 = scatter!(ax, roof_ratio_chamber[325:end], -(underpressure_MPa[325:end]); color = :orange, alpha = 0.4, marker = :star6, markersize =  10, label = "Creep law variations")
    mask_variations = (friction_angle[253:324] .== phi) .& (tectonic_setting[253:324] .== 0.0)
    s3 = scatter!(ax, roof_ratio_chamber[253:324][mask_variations], -(underpressure_MPa[253:324][mask_variations]); color = :blue, alpha = 0.4, marker = :star6, markersize = 10, label = "Ref model Variations")

    if analytical_solution == true
    mask1 = (friction_angle[1:252] .== phi) .& (tectonic_setting[1:252] .== 0.0)
    s2 = scatter!(ax2, roof_ratio_chamber[1:252][mask1], -(underpressure_MPa[1:252][mask1]); color = :black, alpha = 1.0, marker = markers[i], markersize = 8, label = "Models")
    s4 = scatter!(ax2, roof_ratio_chamber[253:324][mask_variations], -(underpressure_MPa[253:324][mask_variations]); color = :blue, alpha = 1.0, marker = :star6, markersize = 8, label = "Ref model Variations")
    s6 = scatter!(ax2, roof_ratio_chamber[325:end], -(underpressure_MPa[325:end]); color = :orange, alpha = 1.0, marker = :star6, markersize = 8, label = "Creep law variations")
    end

    if i == 4 && phi == 30.0
        sc1 = scatter!(ax, [2.0], [(166 + 205)/2], color = cmap[1], markersize = 20, marker = :star5, label = "Katmai")
        rangebars!(ax, [2.0], [166.0], [205.0], color = cmap[1])

        # # Pinatubo: Roof ratio 2.4, Underpressure 265-312 MPa
        sc2 = scatter!(ax, [(2.4 + 3.2) /2], [(265 + 312)/2], color = cmap[3], markersize = 20, marker = :star5, label = "Pinatubo")
        rangebars!(ax, [(2.4 + 3.2) /2], [265.0], [312.0], color = cmap[3])
        rangebars!(ax, [(265 + 312)/2], [2.4], [3.2],direction = :x, color = cmap[3])

        # # Fernandina: Roof ratio 0.31, Underpressure 3-9 MPa
        sc3 = scatter!(ax, [0.31], [(3 + 9)/2], color = cmap[6], markersize = 20, marker = :star5, label = "Fernandina")
        rangebars!(ax, [0.31], [3.0], [9.0], color = cmap[6])

        range       = PolyElement(color = (cmap[7], 0.3), strokecolor = cmap[7])
        scaling     = LineElement(color = cmap[6], linestyle = :solid, linewidth = 3)
        sphere      = LineElement(color = (:black, 0.5), linestyle = :dot, linewidth = 2)
        cylinder    = LineElement(color = (:black, 0.5), linestyle = :dashdot, linewidth = 2)
        models      = [MarkerElement(color = :black, marker = :circle, markersize = 18), MarkerElement(color = :black, marker = :rect, markersize = 18), MarkerElement(color = :black, marker = :diamond, markersize = 18), MarkerElement(color = :black, marker = :utriangle, markersize = 18)]
        creep_laws  = MarkerElement(color = :orange, marker = :star6, markersize = 18)
        ref_var     = MarkerElement(color = :blue, marker = :star6, markersize = 18)
        katmai      = MarkerElement(color = cmap[1], marker = :star5, markersize = 18)
        pinatuno    = MarkerElement(color = cmap[3], marker = :star5, markersize = 18)
        fernandina  = MarkerElement(color = cmap[6], marker = :star5, markersize = 18)
        if analytical_solution == true
            fig[3, 1]   = Legend(fig,
            [range, scaling, sphere, cylinder, models...,
            ref_var, creep_laws, katmai, pinatuno, fernandina],
            ["Range for Depth=5.6-9.45 km", rich("Universal Fit (A = 1.84, B = 2", CairoMakie.superscript("3.2"), ")"), "Hollow Sphere (A = 2)", "Cylinder (A = 2/√3)", "ϕ = 30°", "ϕ = 25°", "ϕ = 20°", "ϕ = 15°",
            "Ref model variations", "Creep law variations", "Katmai", "Pinatubo", "Fernandina"],
            tellheight = true, tellwidth = false,
            orientation = :horizontal, nbanks = 4, fontsize = 22, labelsize = 22
        )
        else
            fig[2, 1]   = Legend(fig,
            [range, scaling, models..., ref_var, creep_laws, katmai, pinatuno, fernandina],
            ["Range for Depth=5.6-9.45 km", rich("Universal Fit (A = 1.84, B = 2", CairoMakie.superscript("3.2"), ")"), "ϕ = 30°", "ϕ = 25°", "ϕ = 20°", "ϕ = 15°",
            "Ref model variations", "Creep law variations", "Katmai", "Pinatubo", "Fernandina"],
            tellheight = true, tellwidth = false,
            orientation = :horizontal, nbanks = 2, fontsize = 22, labelsize = 22
            )
        end
    end


end
ylims!(ax, -50, 325)
xlims!(ax, 0.0, 3.25)
text!(ax, 0.752, -4;  text = L"P_\textrm{under} = (A \ln(BR) \cdot [C \ \cos(\phi) + \bar{P} \ \sin(\phi)]", color = :black, fontsize = 30)

if analytical_solution == true
    panel_label = "a"
    inset_ax = Axis(fig[1,1], width = Relative(0.03), height = Relative(0.1), halign = :left, valign = :top, backgroundcolor = :gray90)
    hidedecorations!(inset_ax); hidespines!(inset_ax)
    text!(inset_ax, 0.5, 0.5, text = panel_label, space = :relative, align = (:center, :center), fontsize = 20, color = :black)

    panel_label = "b"
    inset_ax = Axis(fig[2,1], width = Relative(0.03), height = Relative(0.1), halign = :left, valign = :top, backgroundcolor = :gray90)
    hidedecorations!(inset_ax); hidespines!(inset_ax)
    text!(inset_ax, 0.5, 0.5, text = panel_label, space = :relative, align = (:center, :center), fontsize = 20, color = :black)
    xlims!(ax2, 0.0, 3.25)
    text!(ax2, 0.752, -5;  text = L"P_\textrm{under} = A \cdot \ln(1 + 2R) \cdot [C \ \cos(\phi) + \bar{P} \ \sin(\phi)]", color = :black, fontsize = 24)
end

display(fig)

save("./SmallScaleCaldera/Post_processing/Underpressure_scaling_law_no_analytical_solution.png", fig)
save("./SmallScaleCaldera/Post_processing/Underpressure_scaling_law_no_analytical_solution.pdf", fig)
save("./SmallScaleCaldera/Post_processing/Underpressure_scaling_law_no_analytical_solution.svg", fig)
end
