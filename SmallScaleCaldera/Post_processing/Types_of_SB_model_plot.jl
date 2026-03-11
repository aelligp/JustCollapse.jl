# Underpressure vs roof Ratio SB propagation type
using CairoMakie, XLSX, DataFrames

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

gradient_background = false

let
fig = Figure(size = (1000, 750))
ax = Axis(fig[1, 1];
    aspect = 1.5,
    xlabel = "Roof ratio of chamber (depth / width)",
    ylabel = "Underpressure [MPa]",
    # title = "Onset of caldera collapse vs roof ratio of chamber with Topo added",
    xticks = [0.7, 1.1, collect(0.5:0.5:2.25)...],
    xticklabelsize = 18,
    yticklabelsize = 18,
    xlabelsize = 22,
    ylabelsize = 22,
)

# Smooth horizontal gradient background: left→white at 0.7, white plateau to 1.1, white→right
if gradient_background == true
_grad_vals = [x < 0.75f0  ? (x / 0.75f0) * 0.5f0 :
              x < 1.0f0  ? 0.5f0 :
              0.5f0 + ((x - 1.0f0) / (2.25f0 - 1.0f0)) * 0.5f0
              for x in range(0f0, 2.25f0, 512)]
image!(ax,
    0.0 .. 2.25, 0.0 .. 200.0,
    reshape(_grad_vals, 512, 1) .* ones(Float32, 1, 2);
    colormap = :roma, alpha = 0.750, interpolate = true
)
else
    # add solid colored backgrounds for each region
    cmap_bg = CairoMakie.categorical_colors(:roma10, 10)
    poly!(ax, Rect2f(0.0, 0.0, 0.7, 200.0); color = (cmap_bg[2], 0.4))
    poly!(ax, Rect2f(0.7, 0.0, 0.4, 200.0); color = (cmap_bg[5], 0.4))
    poly!(ax, Rect2f(1.1, 0.0, 1.15, 200.0); color = (cmap_bg[8], 0.4))
end

# cmap = CairoMakie.categorical_colors(:roma10, 10)
# cmap = CairoMakie.categorical_colors(:berlin10, 10)
# cmap = CairoMakie.categorical_colors(:lisbon10, 10)
cmap = CairoMakie.categorical_colors(:grayC, 10)
friction_colors = cmap[1:3:8]
marker_shapes = [:circle, :rect, :diamond]
tectonic_settings_unique = [0.0, 1e-15, -1e-15]
friction_angles_unique = [15.0, 20.0, 25.0, 30.0]
setting_label = ["0.0", "Extension", "Compression"]

for angle in friction_angles_unique
    for (j, setting) in enumerate(tectonic_settings_unique)
        mask = (friction_angle .== angle) .& (tectonic_setting .== setting)
        if any(mask)

            x_vals = roof_ratio_chamber[mask]
            y_vals = abs.(underpressure_MPa[mask])
            valid = .!ismissing.(y_vals) .&& .!ismissing.(x_vals)

            if any(valid)

                ref_mask = x_vals[valid] .≈ 0.68
                upper_mask = x_vals[valid] .≈  1.7
                transition = x_vals[valid] .≈ 0.95
                if any(.!ref_mask) && any(.!upper_mask) && any(.!transition)
                    scatter!(ax, x_vals[valid][.!ref_mask], y_vals[valid][.!ref_mask];
                        color = friction_colors[j],
                        markersize = 18,
                        marker = marker_shapes[j],
                        strokecolor = :white,
                        strokewidth = 0.5,
                        label = "$(setting_label[j])"
                    )
                end
                if any(ref_mask)
                    scatter!(ax, x_vals[valid][ref_mask], y_vals[valid][ref_mask];
                        color = :grey,
                        markersize = 20,
                        marker = :star4,
                        strokecolor = :white,
                        strokewidth = 0.5,
                        label = "Reference model\nType 1"
                    )
                end
                if any(transition)
                    scatter!(ax, x_vals[valid][transition], y_vals[valid][transition];
                        color = :orange,
                        markersize = 20,
                        marker = :star6,
                        strokecolor = :white,
                        strokewidth = 0.5,
                        label = "Transition"
                    )
                end
                if any(upper_mask)
                    scatter!(ax, x_vals[valid][upper_mask], y_vals[valid][upper_mask];
                        color = :purple,
                        markersize = 20,
                        marker = :star5,
                        strokecolor = :white,
                        strokewidth = 0.5,
                        label = "Type 2"
                    )
                end
            end

        end
    end
end
_cmap  = CairoMakie.to_colormap(:broc)
_n     = length(_cmap)
c_down = _cmap[round(Int, 0.10 * _n)]
c_up   = _cmap[round(Int, 0.90 * _n)]

vlines!(ax, 0.7,  color = :black, linestyle = :dash)
vlines!(ax, 1.1,  color = :black, linestyle = :dash)

text!(ax, 0.35,  185; text = "Downward\npropagating", color = :black, fontsize = 24, align = (:center, :top))
text!(ax, 0.9,   185; text = "Transition", color = :black,fontsize = 24, align = (:center, :top))
text!(ax, 1.675, 185; text = "Upward\npropagating", color = :black, fontsize = 24, align = (:center, :top))
xlims!(ax, 0.0, 2.25)
ylims!(ax, 0, 200)
axislegend(ax, position = :rb, merge = true, unique = true, fontsize = 22, labelsize = 22)
display(fig)
save("./SmallScaleCaldera/Post_processing/Types_of_SB_model_plot.png", fig)
save("./SmallScaleCaldera/Post_processing/Types_of_SB_model_plot.svg", fig)
save("./SmallScaleCaldera/Post_processing/Types_of_SB_model_plot.pdf", fig)
end
