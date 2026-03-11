## Diameter vs Erupted volume for different friction angles
using CairoMakie, CSV, XLSX, DataFrames

data = DataFrame(XLSX.readtable("Onset_of_caldera_collapse_CSV.xlsx", "All_models"))
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
eruption_volume = [ismissing(v) ? missing : Float64(v) for v in data[:, 11]]
t_of_through_going_fault = data[:, 12]
underpressure_MPa = data[:, 16]
maximum_undepressure_MPa = data[:, 17]
Shear_band_orientation = data[:, 18]


Caldera_area_km2 = π .* (diameter_caldera ./ 2) .^2 ./ 1e6
radius_Caldera = diameter_caldera ./ 2
total_radius = radius_chamber .* aspect_ratio
density_host_rock = 2700.0 # kg/m^3
gravity = 9.81 # m/s^2

cmap = CairoMakie.categorical_colors(:lipari10, 10)

let

    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1];
        # xlabel = Geshi ? "L/Sc x 10⁴" : "Roof ratio of chamber (depth / width)",
        xlabel = "Caldera diameter [km]",
        # ylabel = "Underpressure at caldera collapse (MPa)",
        ylabel = "Erupted volume (DRE in km³)",
        xlabelsize = 24,
        ylabelsize = 24,
        xticklabelsize = 20,
        yticklabelsize = 20,
        # xscale = log10,
        # yscale = log10,
        xminorticksvisible=true,
        xticks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        xminorticks = [0:0.5:10...],
        # xminorticks=IntervalsBetween(4),
        yminorticksvisible=true,
        yminorticks=IntervalsBetween(5),
    )

    friction_colors = cmap[1:3:8]
    marker_shapes = [:circle, :rect, :diamond]
    # tectonic_settings_unique = 0.0
    tectonic_settings_unique = [0.0, 1e-15, -1e-15]
    friction_angles_unique = [15.0, 20.0, 25.0, 30.0]

    for angle in friction_angles_unique
        for (j, setting) in enumerate(tectonic_settings_unique)
            mask = (friction_angle .== angle) .& (tectonic_setting .== setting)
            if any(mask)
                x_vals = diameter_caldera[mask]./1e3
                y_vals = abs.(eruption_volume[mask])
                valid = .!ismissing.(y_vals) .&& .!ismissing.(x_vals)

                if any(valid)
                    ref_mask = x_vals[valid] .≈ 7.440
                    # upper_ref_mask = x_vals[valid] .≈ 7.510
                    # lower_ref_mask = x_vals[valid] .≈ 6.980
                    if any(.!ref_mask)
                        scatter!(ax, x_vals[valid][.!ref_mask], y_vals[valid][.!ref_mask];
                            # color = friction_colors[2],
                            color = :grey,
                            alpha = 0.5,
                            # color = :black,
                            markersize = 14,
                            # marker = marker_shapes[j],
                            label = "Models"
                        )
                    end
                    if any(ref_mask)
                        scatter!(ax, x_vals[valid][ref_mask], y_vals[valid][ref_mask];
                            color = friction_colors[3],
                            markersize = 18,
                            marker = :star4,
                            strokecolor = friction_colors[3],
                            strokewidth = 2,
                            label = "Reference model"
                        )
                    end
                end
            end
        end
    end


    # # Crater Lake:  geshi 2014 roof ratio 1.0 depth (5-8km) bacon 1983, lipman 1997
    scatter!(ax, [6.5], [55], color = :purple, markersize = 17, marker = :star5, label = "Crater Lake")
    rangebars!(ax, [6.5], [50], [60], color = :purple)
    rangebars!(ax, [55], [5.5], [7.5], direction = :x, color = :purple)
    # Structural diameter ref model
    scatter!(ax, [6.6], [55], color = cmap[8], markersize = 18, marker = :star6, label = "Reference model \nStructural diameter")

    # krakatau: geshi 2014 roof ratio ? (deplus 1995m carey 1996)
    scatter!(ax, [2.4], [12], color = :green, markersize = 17, marker = :star5, label = "Krakatau 1883")
    rangebars!(ax, [2.4], [10], [14], color = :green)
    rangebars!(ax, [12], [2.2], [2.6], direction = :x, color = :green)

    # Katmai: geshi 2014 roof ratio 2.4  depth 5-6km cioni 2008
    scatter!(ax, [2.3], [12], color = :orange, markersize = 17, marker = :star5, label = "Katmai 1912")
    rangebars!(ax, [2.3], [10], [14], color = :orange)
    rangebars!(ax, [12], [2.1], [2.5], direction = :x, color = :orange)

    # Aniakchak: Miller and Smith 1987, dreher
    # scatter!(ax, [10], [27], color = :brown, markersize = 17, marker = :star5, label = "Aniakchak")
    # rangebars!(ax, [10], [15], [25], color = :brown)
    # rangebars!(ax, [20], [6.5], [7.5], direction = :x, color = :brown)

    # Ksudach Kamchatka Braitseva et al. (1996) 240AD Caldera
    scatter!(ax, [5], [8], color = :pink, markersize = 17, marker = :star5, label = "Ksudach 240 AD")
    rangebars!(ax, [5], [7], [9], color = :pink)
    rangebars!(ax, [8], [4], [6.5], direction = :x, color = :pink)

    # tambora 1815 Foden (1986), Gertisser(2012)
    scatter!(ax, [6], [33.2], color = :red, markersize = 17, marker = :star5, label = "Tambora 1815")
    rangebars!(ax, [6], [23.2], [43.2], color = :red)
    # rangebars!(ax, [33.2], [4.5], [5.5], direction = :x, color = :red)

    # Pinatubo: geshi 2014
    scatter!(ax, [2.2], [4.5], color = cmap[5], markersize = 17, marker = :star5, label = "Pinatubo 1991")
    rangebars!(ax, [2.2], [3.7], [5.3], color = cmap[5])
    rangebars!(ax, [4.5], [2.0], [2.4], direction = :x, color = cmap[5])

    xlims!(ax, 1e0, 10)
    ylims!(ax, 0e0, 1.5*10^2)

    # fig[1, 1] = Legend(fig, ax, position = :lt, merge = true, unique = true)
    axislegend(ax, position = :lt, merge = true, unique = true, labelsize = 20)
    display(fig)

    save("./SmallScaleCaldera/Post_processing/Diameter_vs_Volume.png", fig)
    save("./SmallScaleCaldera/Post_processing/Diameter_vs_Volume.svg", fig)
    save("./SmallScaleCaldera/Post_processing/Diameter_vs_Volume.pdf", fig)
end
