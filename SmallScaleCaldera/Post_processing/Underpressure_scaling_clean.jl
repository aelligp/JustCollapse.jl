using CairoMakie, CSV, XLSX, DataFrames, LsqFit


function compute_perimeter_length_caldera(r_caldera)
    return 2 * π * r_caldera
end

function compute_basal_area_caldera(r_caldera)
    return π * r_caldera^2
end

"""
    P_under_Geshi_Eq_1()

Calculate the underpressure required for caldera collapse based on Geshi 2024. Eq. (1).

# Arguments
- `friction_angle::Float64`: Friction angle of the host rock in degrees. Will be converted into friction coefficient μ = tand(friction_angle).
- `density_host_rock::Float64`: Density of the host rock in kg/m^3.
- `gravity::Float64`: Gravitational acceleration in m/s^2.
- `H::Float64`: Depth of the magma chamber in meters.
- `L::Float64`: perimeter length of the caldera block
- `Sc::Float64`: Basal area of the caldera block
"""
function P_under_Geshi_Eq_1(friction_coeff, density_host_rock, gravity, H, L, Sc)
    # μ = tand(friction_angle)
   P_u = @.  1/2 * friction_coeff * density_host_rock * gravity * (L / Sc) * H^2
    return P_u
end


"""
    P_under(friction_angle, r_caldera, density_host_rock, depth, gravity)

Calculate the underpressure required for caldera collapse based on Geshi 2024. Eq. (6).
Assuming a circular caldera radius r.

# Arguments
- `friction_angle::Float64`: Friction angle of the host rock in degrees.
- `r_caldera::Float64`: Radius of the caldera in meters.
- `density_host_rock::Float64`: Density of the host rock in kg/m^
- `depth::Float64`: Depth of the magma chamber in meters.
- `gravity::Float64`: Gravitational acceleration in m/s^2.
# Returns
- `P_u::Float64`: Underpressure in Pascal.
"""
function P_under(friction_coeff, r_caldera, density_host_rock, depth, gravity)
    # μ = tand(friction_angle)

    P_u = friction_coeff * density_host_rock * gravity * (depth^2 / r_caldera)
    return P_u
end

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

let
    Geshi = false
    # Storage for coefficients across friction angles
    all_slopes = Float64[]
    all_intercepts = Float64[]
    all_angles = Float64[]

    for k in [15.0, 20.0, 25.0, 30.0]

        fig = Figure(size = (800, 600))
        ax = Axis(fig[1, 1];
            xlabel = Geshi ? "L/Sc x 10⁴" : "Roof ratio of chamber (depth / width)",
            ylabel = "Underpressure [MPa]",
            # title = "Onset of caldera collapse vs roof ratio of chamber with Topo added",
        )

        friction_angles_unique = k
        # friction_angles_unique = 30.0


        cmap = CairoMakie.categorical_colors(:lipari10, 10)
        friction_colors = cmap[1:3:8]
        marker_shapes = [:circle, :rect, :diamond]
        # tectonic_settings_unique = 0.0
        tectonic_settings_unique = [0.0, 1e-15, -1e-15]

        # Compute L_over_Sc for this loop
        L = compute_perimeter_length_caldera.(diameter_caldera ./ 2)
        Sc = compute_basal_area_caldera.(diameter_caldera ./ 2)
        P_u_Geshi_Eq1 = P_under_Geshi_Eq_1.(friction_coeff, density_host_rock, gravity, depth_chamber.*1e3, L, Sc)
        P_u_Geshi_Eq1_MPa = P_u_Geshi_Eq1 ./ 1e6
        P_u_Geshi_Eq5 = P_under.( friction_coeff, radius_Caldera, density_host_rock, (depth_chamber.*1e3), gravity)
        P_u_Geshi_Eq5_MPa = P_u_Geshi_Eq5 ./ 1e6
        # L_over_Sc = compute_perimeter_length_caldera.(diameter_caldera ./ 2) ./ compute_basal_area_caldera.(diameter_caldera ./ 2)

        for (i, angle) in enumerate(friction_angles_unique)
            for (j, setting) in enumerate(tectonic_settings_unique)
                mask = (friction_angle .== angle) .& (tectonic_setting .== setting)
                if any(mask)
                    x_vals = Geshi ? L[mask]./Sc[mask].*1e4 : roof_ratio_chamber[mask]
                    y_vals = abs.(underpressure_MPa[mask])
                    # y_vals = (underpressure_MPa[mask])
                    valid = .!ismissing.(y_vals) .&& .!ismissing.(x_vals)



                    if any(valid)
                        # ref_mask = x_vals[valid] .≈ 0.7142857142857143
                        ref_mask = x_vals[valid] .≈ 0.691428571

                        if any(.!ref_mask)
                            scatter!(ax, x_vals[valid][.!ref_mask], y_vals[valid][.!ref_mask];
                                color = friction_colors[j],
                                markersize = 12,
                                marker = marker_shapes[j],
                                label = "$(Int(angle))°, ε̇: $(setting)"
                            )

                            if Geshi == true
                            scatter!(ax, x_vals[valid][.!ref_mask], P_u_Geshi_Eq1_MPa[mask][valid][.!ref_mask];
                            # scatter!(ax, x_vals[valid][.!ref_mask], P_u_Geshi_Eq5_MPa[mask][valid][.!ref_mask];
                                # color = friction_colors[j],
                                color = :blue,
                                markersize = 8,
                                marker = marker_shapes[j],
                                label = "Geshi Eq. 1"
                            )
                            end
                        end

                        if any(ref_mask)
                            scatter!(ax, x_vals[valid][ref_mask], y_vals[valid][ref_mask];
                                color = friction_colors[j],
                                markersize = 16,
                                marker = :star4,
                                strokecolor = :grey,
                                strokewidth = 2,
                                label = "reference model"
                            )
                            if Geshi == true
                            scatter!(ax, x_vals[valid][ref_mask], P_u_Geshi_Eq1_MPa[mask][valid][ref_mask];
                            # scatter!(ax, x_vals[valid][ref_mask], P_u_Geshi_Eq5_MPa[mask][valid][ref_mask];
                                color = friction_colors[j],
                                markersize = 8,
                                marker = :star4,
                                strokecolor = :grey,
                                label = "Geshi Eq. 1"
                            )
                            end
                        end
                    end
                end
            end
        end


        # underpressure_scaling = @. (2700 * 9.81 * (depth_chamber*1e3) * roof_ratio_chamber * tand.(30)) ./ 1e6 # in MPa

        # scatter!(ax, roof_ratio_chamber, underpressure_scaling;
        # color = :purple,
        # label = "P_u ~ depth * roof_ratio * tan(friction_angle)"
        # )

        # Katmai: Roof ratio 2.0, Underpressure 166-205 MPa
        scatter!(ax, [2.0], [(166 + 205)/2], color = :red, markersize = 15, marker = :star5, label = "Katmai")
        rangebars!(ax, [2.0], [166.0], [205.0], color = :red)

        # # Pinatubo: Roof ratio 2.4, Underpressure 265-312 MPa
        # scatter!(ax, [2.4], [(265 + 312)/2], color = :blue, markersize = 15, marker = :star5, label = "Pinatubo")
        # rangebars!(ax, [2.4], [265.0], [312.0], color = :blue)


        Geshi ? xlims!(ax, 1, 10) : xlims!(ax, 0.1, 2.5)
        # ylims!(ax, -20, 350 )
        # axislegend(ax, "Friction angle", position = :lb)

        # -------------------------------------------------------------------------
        # New Figure for Geshi Eq. 1 Contours
        # -------------------------------------------------------------------------
        L_Sc_ratio = exp10.(range(log10(1e-4), log10(50e-4), length=100))
        depths_km = exp10.(range(log10(1), log10(15), length=100))

        P_u_eq1_matrix = zeros(length(L_Sc_ratio), length(depths_km))
        phi_val = tand(friction_angles_unique)
        rho_val = 2700.0
        g_val = 9.81

        Crater_Lake_L = compute_perimeter_length_caldera(6500/2) # m
        Crater_Lake_Sc = compute_basal_area_caldera(6500/2) # m^2
        Crater_Lake_L_Sc = Crater_Lake_L / Crater_Lake_Sc

        Crater_Lake_depth = 4000.0 # m


        println("Crater Lake L/Sc (x10⁴): ", Crater_Lake_L_Sc .*1e4)

        for (ix, lsc) in enumerate(L_Sc_ratio)
            for (iy, d_km) in enumerate(depths_km)
                d_m = d_km * 1e3
                # Eq 1: 1/2 * mu * rho * g * (L/Sc) * H^2
                # Result in MPa
                P_u_eq1_matrix[ix, iy] = 0.5 * phi_val * rho_val * g_val * lsc * d_m^2 / 1e6
            end
        end

        fig1 = if Geshi == true
            fig1 = Figure(size = (900, 600))
            ax1 = Axis(fig1[1, 1],
                xlabel = "L/Sc ratio (×10⁻⁴)",
                ylabel = "Depth to magma chamber center (km)",
                title = "Geshi Eq. 1: Underpressure with k=$(Int(k))° and ε_bg=$(tectonic_settings_unique)",
                xgridvisible = true,
                ygridvisible = true,
                # xreversed = true,
                xscale = log10,
                yscale = log10,
                titlesize = 25,
                xlabelsize = 20,
                ylabelsize = 20,
                xminorticksvisible=true,
                xminorticks=IntervalsBetween(9),
                yminorticksvisible=true,
                yminorticks=IntervalsBetween(9),
            )
            # xlims!(ax1, minimum(L_Sc_ratio), maximum(L_Sc_ratio))
            # ylims!(ax1, minimum(depths_km), maximum(depths_km))

            # contour!(ax1, L_Sc_ratio, depths_km, P_u_eq1_matrix;
            contour!(ax1, L_Sc_ratio .* 1e4, depths_km, P_u_eq1_matrix;
                levels = [10, 20, 30, 40, 50, 60, 70, 80,  90, 100, 200],
                linewidth = 1.5,
                labels = true,
                labelsize = 12,
            )

            # Plot Data Scatters
            # Using filtered data for current friction angle
            for setting in tectonic_settings_unique
                mask_d = (friction_angle .== k) .& (tectonic_setting .== setting)
                if any(mask_d)
                    # data points
                    lsc_data = compute_perimeter_length_caldera.(diameter_caldera[mask_d] ./ 2) ./ compute_basal_area_caldera.(diameter_caldera[mask_d] ./ 2)
                    d_km_data = depth_chamber[mask_d] # Assuming depth_chamber is in km
                    pu_data = abs.(underpressure_MPa[mask_d])
                    valid_d = .!ismissing.(pu_data)

                    if any(valid_d)
                        scatter!(ax1, lsc_data[valid_d] .*1e4, d_km_data[valid_d];
                            color = pu_data[valid_d],
                            colormap = :viridis,
                            colorrange = (minimum(P_u_eq1_matrix), maximum(P_u_eq1_matrix)), # Match contour range ideally
                            markersize = 15,
                            strokecolor = :black,
                            strokewidth = 1,
                            label = "Simulations (Color=Pu)"
                        )
                    end
                end
            end

            display(fig1)
            scatter!(ax1, [Crater_Lake_L_Sc .*1e4], [Crater_Lake_depth /1e3];
                color = :red,
                markersize = 20,
                marker = :star4,
                strokecolor = :black,
                strokewidth = 2,
                label = "Crater Lake (estimated)"
            )

            # Colorbar(fig1[1, 2], limits = extrema(P_u_eq1_matrix), colormap=:viridis, label="Underpressure (MPa)")
            # xlims!(ax1, 1e0, 1e1)
            ylims!(ax1, 1e0, 1e1)
            axislegend(ax1, position = :rt)
            display(fig1)
        end

        # =====================================================================
        # Log-form Drucker-Prager fit:
        #   ΔP [MPa] = (a·ln(R) + b) · τ_DP_eff(H)
        # where τ_DP_eff = C·cos(φ) + (ρgH/2)/1e6 · sin(φ)  [MPa]
        # H = depth to chamber [m]. Uses lithostatic at mid-depth as the
        # effective confining pressure (stress concentration around the
        # cylindrical chamber compensates the low far-field K₀ = 1/3).
        # =====================================================================
        Roof_ratio_lines = collect(0.1:0.01:3.0)
        C_rock = 10.0   # cohesion in MPa
        ρ = 2700.0      # kg/m³
        g = 9.81        # m/s²

        # Collect data for fitting
        x_fit = Float64[]       # roof ratio R
        depth_fit = Float64[]   # depth H [km]
        diam_fit = Float64[]    # diameter D [m]
        y_fit = Float64[]       # |ΔP| [MPa]

        for angle in friction_angles_unique
            for setting in tectonic_settings_unique
                mask = (friction_angle .== angle) .& (tectonic_setting .== setting)
                if any(mask)
                    x_vals = roof_ratio_chamber[mask]
                    depth_vals = depth_chamber[mask]        # km
                    diam_vals = diameter_caldera[mask]       # m
                    y_vals = abs.(underpressure_MPa[mask])
                    valid = .!ismissing.(y_vals) .& .!ismissing.(x_vals) .& .!ismissing.(diam_vals)

                    if any(valid)
                        Base.append!(x_fit, Float64.(x_vals[valid]))
                        Base.append!(depth_fit, Float64.(depth_vals[valid]))
                        Base.append!(diam_fit, Float64.(diam_vals[valid]))
                        Base.append!(y_fit, Float64.(y_vals[valid]))
                    end
                end
            end
        end

        # Define log-DP model and fit
        if length(x_fit) > 2
            # X[:,1] = R (roof ratio), X[:,2] = D [m], X[:,3] = H [km]
            X_data = hcat(x_fit, diam_fit, depth_fit)

            # τ_DP_eff = C·cos(φ) + (ρgH/2)/1e6 · sin(φ)  [MPa]
            tau_DP_eff(H_km, phi) = C_rock * cosd(phi) + (ρ * g * H_km * 1e3 / 2.0) / 1e6 * sind(phi)

            # Model: ΔP = (a·ln(R) + b) · τ_DP_eff(H, φ)
            # p[1] = a (geometry/log slope), p[2] = b (geometry/intercept at R=1)
            model_log_DP(X, p) = begin
                R = X[:, 1]
                H_km = X[:, 3]
                tau = tau_DP_eff.(H_km, k)
                return (p[1] .* log.(R) .+ p[2]) .* tau
            end
            p0 = [2.0, 2.5]

            try
                fit = curve_fit(model_log_DP, X_data, y_fit, p0)
                params = coef(fit)
                a_fit = params[1]
                b_fit = params[2]
                sig = stderror(fit)
                pred = model_log_DP(X_data, params)
                ss_res = sum((y_fit .- pred).^2)
                ss_tot = sum((y_fit .- sum(y_fit)/length(y_fit)).^2)
                r_sq = 1.0 - ss_res / ss_tot

                mean_H_km = sum(depth_fit) / length(depth_fit)
                tau_mean = tau_DP_eff(mean_H_km, k)

                println("Friction angle: $(Int(k))°")
                println("  a (log slope):   $(round(a_fit, digits=3)) ± $(round(sig[1], digits=3))")
                println("  b (R=1 factor):  $(round(b_fit, digits=3)) ± $(round(sig[2], digits=3))")
                println("  R² = $(round(r_sq, digits=4))")
                println("  τ_DP_eff(H̄=$(round(mean_H_km,digits=1))km) = $(round(tau_mean, digits=1)) MPa")
                println("  Equation: ΔP = ($(round(a_fit,digits=2))·ln(R) + $(round(b_fit,digits=2))) · [C·cos(φ) + ρgH/(2e6)·sin(φ)]")
                println()

                # Store coefficients for master plot
                push!(all_slopes, a_fit)
                push!(all_intercepts, b_fit)
                push!(all_angles, k)

                # Plot fit curve at mean depth
                scaling_log = (a_fit .* log.(Roof_ratio_lines) .+ b_fit) .* tau_mean
                lines!(ax, Roof_ratio_lines, scaling_log;
                    color = :black,
                    linestyle = :dash,
                    label = "DP-log fit (R²=$(round(r_sq,digits=3)))"
                )

                # Geshi analytical for comparison
                # mean_diam = sum(diam_fit) / length(diam_fit)
                # geshi_curve = 2.0 .* tand(k) .* (ρ * g * mean_diam / 1e6) .* Roof_ratio_lines.^2
                # lines!(ax, Roof_ratio_lines, geshi_curve;
                #     color = :red,
                #     linestyle = :dot,
                #     label = "Geshi Eq. 6 (D̄=$(round(mean_diam/1e3, digits=1)) km)"
                # )

            catch e
                @warn "Fitting failed for k=$k: $e"
            end
        end

        scaling_log_universal = (1.84 .* log.(Roof_ratio_lines) .+ 2.22) .* tau_DP_eff(5.8, k)
                lines!(ax, Roof_ratio_lines, scaling_log_universal;
                    color = :blue,
                    linestyle = :dot,
                    label = "Universal Scaling Law H = 5.8km"
                )


        fig[1, 2] = Legend(fig, ax, "Friction angle", position = :lt, merge=true, unique = true)
        # axislegend(ax, "Friction angle", position = :lt)
        display(fig)
        # save("./SmallScaleCaldera/Post_processing/Onset_caldera_collapse_vs_roof_ratio_chamber_with_Topo_added_$(Int64(k))fa_all_regimes.png", fig)
        # save("./SmallScaleCaldera/Post_processing/Onset_caldera_collapse_vs_roof_ratio_chamber_with_Topo_added_$(Int64(k))fa_all_regimes.svg", fig)
        # save("./SmallScaleCaldera/Onset_caldera_collapse_vs_roof_ratio_chamber_with_Topo_added_$(Int64(k))fa_all_regimes.png", fig)
    end

    # =====================================================================
    # Master plot: Log-DP scaling for all friction angles
    #   ΔP = (a·ln(R) + b) · τ_DP_eff(H)
    # where τ_DP_eff = C·cos(φ) + (ρgH/2)/1e6 · sin(φ)
    # =====================================================================
    C_rock = 10.0   # MPa
    ρ = 2700.0      # kg/m³
    g = 9.81        # m/s²
    tau_DP_master(H_km, phi) = C_rock * cosd(phi) + (ρ * g * H_km * 1e3 / 2.0) / 1e6 * sind(phi)

    valid_depths = collect(skipmissing(depth_chamber))
    mean_H_km_all = sum(valid_depths) / length(valid_depths)
    Roof_ratio_lines = collect(0.1:0.01:3.0)

    mean_a = sum(all_slopes) / length(all_slopes)
    mean_b = sum(all_intercepts) / length(all_intercepts)

    println("\n===== Summary of DP-log fitted coefficients =====")
    println("  Mean depth: $(round(mean_H_km_all, digits=1)) km")
    for (i, phi) in enumerate(all_angles)
        tau = tau_DP_master(mean_H_km_all, phi)
        println("  φ=$(Int(phi))°: a=$(round(all_slopes[i],digits=3)), b=$(round(all_intercepts[i],digits=3)), τ_DP_eff=$(round(tau,digits=1)) MPa")
    end
    println("  Mean a = $(round(mean_a, digits=3))")
    println("  Mean b = $(round(mean_b, digits=3))")
    println("=================================================")
    println("  Universal equation:")
    println("    ΔP = ($(round(mean_a,digits=2))·ln(R) + $(round(mean_b,digits=2))) · [C·cos(φ) + ρgH/(2×10⁶)·sin(φ)]")
    println("=========================================\n")

    fig2 = Figure(size = (900, 600))
    ax2 = Axis(fig2[1, 1];
        xlabel = "Roof ratio R = H/D",
        ylabel = "Underpressure ΔP [MPa]",
        title = "ΔP = (a·ln(R) + b) · [C·cos(φ) + ρgH/(2e6)·sin(φ)]  (H̄=$(round(mean_H_km_all,digits=1)) km)",
        xticks = 0.5:0.5:3.0,
    )
    cmap = CairoMakie.categorical_colors(:lipari10, 10)
    line_color = cmap[1:2:9]
    for (i, phi) in enumerate(all_angles)
        tau = tau_DP_master(mean_H_km_all, phi)

        # Per-angle fit
        curve_fit_i = (all_slopes[i] .* log.(Roof_ratio_lines) .+ all_intercepts[i]) .* tau
        lines!(ax2, Roof_ratio_lines, curve_fit_i;
            color = line_color[i], linestyle = :solid, linewidth = 2,
            label = "φ=$(Int(phi))° (a=$(round(all_slopes[i],digits=2)), b=$(round(all_intercepts[i],digits=2)))")

        # Universal fit (mean a, b)
        curve_universal = (mean_a .* log.(Roof_ratio_lines) .+ mean_b) .* tau
        lines!(ax2, Roof_ratio_lines, curve_universal;
            color = line_color[i], linestyle = :dot, linewidth = 1,
            label = "φ=$(Int(phi))° universal")
    end

    scatter!(ax2, [2.0], [(166 + 205)/2], color = :red, markersize = 15, marker = :star5, label = "Katmai")
    rangebars!(ax2, [2.0], [166.0], [205.0], color = :red)

    scatter!(ax2, [2.4], [185.37], color= :blue, markersize = 15, marker = :star5, label = "Pinatubo (mean)")
    rangebars!(ax2, [2.4],[134.6], [236.1], color = :blue)

    ylims!(ax2, 0, 250)
    xlims!(ax2, 0.1, 2.5)
    fig2[1, 2] = Legend(fig2, ax2, "Model", position = :lt, merge = true, unique = true)
    display(fig2)
    # save("./SmallScaleCaldera/Post_processing/Scaling_DP_log_fit.png", fig2)
    save("./SmallScaleCaldera/Post_processing/Scaling_DP_log_fit.pdf", fig2)
end

# volume vs caldera diameter
let
    for k in [15.0, 20.0, 25.0, 30.0]

        fig = Figure(size = (800, 600))
        ax = Axis(fig[1, 1];
            # xlabel = Geshi ? "L/Sc x 10⁴" : "Roof ratio of chamber (depth / width)",
            xlabel = "Caldera diameter [km]",
            # ylabel = "Underpressure at caldera collapse (MPa)",
            ylabel = "Erupted volume (km³)",
            xscale = log10,
            yscale = log10,
            xminorticksvisible=true,
            xminorticks=IntervalsBetween(9),
            yminorticksvisible=true,
            yminorticks=IntervalsBetween(9),
        )

        friction_angles_unique = k

        cmap = CairoMakie.categorical_colors(:lipari10, 10)
        friction_colors = cmap[1:3:8]
        marker_shapes = [:circle, :rect, :diamond]
        # tectonic_settings_unique = 0.0
        tectonic_settings_unique = [0.0, 1e-15, -1e-15]

        for (i, angle) in enumerate(friction_angles_unique)
            for (j, setting) in enumerate(tectonic_settings_unique)
                mask = (friction_angle .== angle) .& (tectonic_setting .== setting)
                if any(mask)
                    x_vals = diameter_caldera[mask]./1e3
                    y_vals = abs.(eruption_volume[mask])
                    valid = .!ismissing.(y_vals) .&& .!ismissing.(x_vals)

                    if any(valid)
                        ref_mask = x_vals[valid] .≈ 0.691428571

                        if any(.!ref_mask)
                            scatter!(ax, x_vals[valid][.!ref_mask], y_vals[valid][.!ref_mask];
                                color = friction_colors[j],
                                markersize = 12,
                                marker = marker_shapes[j],
                                label = "$(Int(angle))°, ε̇: $(setting)"
                            )
                        end

                        if any(ref_mask)
                            scatter!(ax, x_vals[valid][ref_mask], y_vals[valid][ref_mask];
                                color = friction_colors[j],
                                markersize = 16,
                                marker = :star4,
                                strokecolor = :grey,
                                strokewidth = 2,
                                label = "reference model"
                            )
                        end
                    end
                end
            end
        end




        # Katmai: Roof ratio 2.0, Underpressure 166-205 MPa
        # scatter!(ax, [2.0], [(166 + 205)/2], color = :red, markersize = 15, marker = :star5, label = "Katmai")
        # rangebars!(ax, [2.0], [166.0], [205.0], color = :red)

        # # Pinatubo: Roof ratio 2.4, Underpressure 265-312 MPa
        # scatter!(ax, [2.4], [(265 + 312)/2], color = :blue, markersize = 15, marker = :star5, label = "Pinatubo")
        # rangebars!(ax, [2.4], [265.0], [312.0], color = :blue)

        # # # Pinatubo: Roof ratio 2.4, Underpressure 265-312 MPa
        # scatter!(ax, [2.2], [4.5], color = :blue, markersize = 15, marker = :star5, label = "Pinatubo")
        # rangebars!(ax, [2.2], [3.7], [5.3], color = :blue)
        # rangebars!(ax, [4.5], [2.0], [2.4], direction = :x, color = :blue)

        # # Crater Lake:  geshi 2013 roof ratio 1.0 depth (5-8km) bacon 1983, lipman 1997
        scatter!(ax, [6.5], [55], color = :purple, markersize = 15, marker = :star5, label = "Crater Lake")
        rangebars!(ax, [6.5], [50], [60], color = :purple)
        rangebars!(ax, [55], [5.5], [7.5], direction = :x, color = :purple)

        # krakatau: geshi 2013 roof ratio ? (deplus 1995m carey 1996)
        scatter!(ax, [2.4], [12], color = :green, markersize = 15, marker = :star5, label = "Krakatau")
        rangebars!(ax, [2.4], [10], [14], color = :green)
        rangebars!(ax, [12], [2.2], [2.6], direction = :x, color = :green)

        # Katmai: geshi 2013 roof ratio 2.4  depth 5-6km cioni 2008
        scatter!(ax, [2.3], [12], color = :orange, markersize = 15, marker = :star5, label = "Katmai")
        rangebars!(ax, [2.3], [10], [14], color = :orange)
        rangebars!(ax, [12], [2.1], [2.5], direction = :x, color = :orange)

        # Aniakchak: Bacon and others (2014), Miller and Smith 1998/1999
        scatter!(ax, [7], [20], color = :brown, markersize = 15, marker = :star5, label = "Aniakchak")
        rangebars!(ax, [7], [15], [25], color = :brown)
        rangebars!(ax, [20], [6.5], [7.5], direction = :x, color = :brown)

        # Ksudach Kamchatka Braitseva et al. (1996) 240AD Caldera
        scatter!(ax, [4], [8], color = :pink, markersize = 15, marker = :star5, label = "Ksudach")
        rangebars!(ax, [4], [7], [9], color = :pink)
        rangebars!(ax, [8], [3], [5], direction = :x, color = :pink)

        # tambora 1815 Foden (1986) after geshi 2013
        scatter!(ax, [5], [33.2], color = :red, markersize = 15, marker = :star5, label = "Tambora")
        rangebars!(ax, [5], [23.2], [43.2], color = :red)
        rangebars!(ax, [33.2], [4.5], [5.5], direction = :x, color = :red)


        # # Vesuvuis: geshi 2013 roof ratio 2.4  depth 5-6km cioni 2008
        # scatter!(ax, [2.3], [1.4], color = :orange, markersize = 15, marker = :star5, label = "Vesuvius")
        # rangebars!(ax, [2.3], [1.2], [1.6], color = :orange)
        # rangebars!(ax, [1.4], [2.1], [2.5], direction = :x, color = :orange)


        # Geshi ? xlims!(ax, 1, 10) : xlims!(ax, 0.1, 3.0)
        # [:black, :blue, :green, :orange, :purple, :brown, :pink]
        # xlims!(ax, 1e0, 1e1)
        ylims!(ax, 1e0, 1.5*10^2)
        # axislegend(ax, "Friction angle", position = :lb)

        fig[1, 2] = Legend(fig, ax, "Friction angle", position = :lt)
        # axislegend(ax, "Friction angle", position = :lt)
        display(fig)

        # save("./SmallScaleCaldera/Post_processing/Diameter_vs_Volume_fa_$(Int(k)).png", fig)
        # save("./SmallScaleCaldera/Post_processing/Diameter_vs_Volume_fa_$(Int(k)).svg", fig)
        # save("./SmallScaleCaldera/Post_processing/Diameter_vs_Volume_fa_$(Int(k)).pdf", fig)
    end
end

let

    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1];
        # xlabel = Geshi ? "L/Sc x 10⁴" : "Roof ratio of chamber (depth / width)",
        xlabel = "Caldera diameter [km]",
        # ylabel = "Underpressure at caldera collapse (MPa)",
        ylabel = "Erupted volume (km³)",
        # xscale = log10,
        # yscale = log10,
        xminorticksvisible=true,
        xminorticks=IntervalsBetween(9),
        yminorticksvisible=true,
        yminorticks=IntervalsBetween(9),
    )


    cmap = CairoMakie.categorical_colors(:lipari10, 10)
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
                    # Note: Check if 0.691 is the correct reference value for DIAMETER (km)
                    # or if this was copied from the roof_ratio plot.
                    ref_mask = x_vals[valid] .≈ 0.691428571

                    if any(.!ref_mask)
                        scatter!(ax, x_vals[valid][.!ref_mask], y_vals[valid][.!ref_mask];
                            color = friction_colors[j],
                            markersize = 12,
                            marker = marker_shapes[j],
                            label = "ε̇: $(setting)"
                        )
                    end

                    if any(ref_mask)
                        scatter!(ax, x_vals[valid][ref_mask], y_vals[valid][ref_mask];
                            color = friction_colors[j],
                            markersize = 16,
                            marker = :star4,
                            strokecolor = :grey,
                            strokewidth = 2,
                            label = "reference model"
                        )
                    end
                end
            end
        end
    end




    # Katmai: Roof ratio 2.0, Underpressure 166-205 MPa
    # scatter!(ax, [2.0], [(166 + 205)/2], color = :red, markersize = 15, marker = :star5, label = "Katmai")
    # rangebars!(ax, [2.0], [166.0], [205.0], color = :red)

    # # Pinatubo: Roof ratio 2.4, Underpressure 265-312 MPa
    # scatter!(ax, [2.4], [(265 + 312)/2], color = :blue, markersize = 15, marker = :star5, label = "Pinatubo")
    # rangebars!(ax, [2.4], [265.0], [312.0], color = :blue)

    # # # Pinatubo: Roof ratio 2.4, Underpressure 265-312 MPa
    # scatter!(ax, [2.2], [4.5], color = :blue, markersize = 15, marker = :star5, label = "Pinatubo")
    # rangebars!(ax, [2.2], [3.7], [5.3], color = :blue)
    # rangebars!(ax, [4.5], [2.0], [2.4], direction = :x, color = :blue)

    # # Crater Lake:  geshi 2013 roof ratio 1.0 depth (5-8km) bacon 1983, lipman 1997
    scatter!(ax, [6.5], [55], color = :purple, markersize = 15, marker = :star5, label = "Crater Lake")
    rangebars!(ax, [6.5], [50], [60], color = :purple)
    rangebars!(ax, [55], [5.5], [7.5], direction = :x, color = :purple)

    # krakatau: geshi 2013 roof ratio ? (deplus 1995m carey 1996)
    scatter!(ax, [2.4], [12], color = :green, markersize = 15, marker = :star5, label = "Krakatau")
    rangebars!(ax, [2.4], [10], [14], color = :green)
    rangebars!(ax, [12], [2.2], [2.6], direction = :x, color = :green)

    # Katmai: geshi 2013 roof ratio 2.4  depth 5-6km cioni 2008
    scatter!(ax, [2.3], [12], color = :orange, markersize = 15, marker = :star5, label = "Katmai")
    rangebars!(ax, [2.3], [10], [14], color = :orange)
    rangebars!(ax, [12], [2.1], [2.5], direction = :x, color = :orange)

    # Aniakchak: Bacon and others (2014), Miller and Smith 1998/1999
    scatter!(ax, [7], [20], color = :brown, markersize = 15, marker = :star5, label = "Aniakchak")
    rangebars!(ax, [7], [15], [25], color = :brown)
    rangebars!(ax, [20], [6.5], [7.5], direction = :x, color = :brown)

    # Ksudach Kamchatka Braitseva et al. (1996) 240AD Caldera
    scatter!(ax, [4], [8], color = :pink, markersize = 15, marker = :star5, label = "Ksudach")
    rangebars!(ax, [4], [7], [9], color = :pink)
    rangebars!(ax, [8], [3], [5], direction = :x, color = :pink)

    # tambora 1815 Foden (1986) after geshi 2013
    scatter!(ax, [5], [33.2], color = :red, markersize = 15, marker = :star5, label = "Tambora")
    rangebars!(ax, [5], [23.2], [43.2], color = :red)
    rangebars!(ax, [33.2], [4.5], [5.5], direction = :x, color = :red)


    # # Vesuvuis: geshi 2013 roof ratio 2.4  depth 5-6km cioni 2008
    # scatter!(ax, [2.3], [1.4], color = :orange, markersize = 15, marker = :star5, label = "Vesuvius")
    # rangebars!(ax, [2.3], [1.2], [1.6], color = :orange)
    # rangebars!(ax, [1.4], [2.1], [2.5], direction = :x, color = :orange)


    # Geshi ? xlims!(ax, 1, 10) : xlims!(ax, 0.1, 3.0)
    # [:black, :blue, :green, :orange, :purple, :brown, :pink]
    xlims!(ax, 1e0, 1e1)
    ylims!(ax, 1e0, 1.5*10^2)
    # axislegend(ax, "Friction angle", position = :lb)

    fig[1, 2] = Legend(fig, ax, "Friction angle", position = :lt, merge = true, unique = true)
    # axislegend(ax, "Friction angle", position = :lt)
    display(fig)

    # save("./SmallScaleCaldera/Post_processing/Diameter_vs_Volume_fa_$(Int(k)).png", fig)
    # save("./SmallScaleCaldera/Post_processing/Diameter_vs_Volume_fa_$(Int(k)).svg", fig)
    # save("./SmallScaleCaldera/Post_processing/Diameter_vs_Volume_fa_$(Int(k)).pdf", fig)
end


# underpressure vs caldera diameter
let

    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1];
        # xlabel = Geshi ? "L/Sc x 10⁴" : "Roof ratio of chamber (depth / width)",
        xlabel = "Caldera diameter [km]",
        ylabel = "Underpressure at caldera collapse (Pa)",
        # ylabel = "Erupted volume (km³)",
        xscale = log10,
        yscale = log10,
        xminorticksvisible=true,
        xminorticks=IntervalsBetween(9),
        yminorticksvisible=true,
        yminorticks=IntervalsBetween(9),
    )


    cmap = CairoMakie.categorical_colors(:lipari10, 10)
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
                y_vals = abs.(underpressure_MPa[mask].*1e6)
                valid = .!ismissing.(y_vals) .&& .!ismissing.(x_vals)

                if any(valid)
                    # Note: Check if 0.691 is the correct reference value for DIAMETER (km)
                    # or if this was copied from the roof_ratio plot.
                    ref_mask = x_vals[valid] .≈ 0.691428571

                    if any(.!ref_mask)
                        scatter!(ax, x_vals[valid][.!ref_mask], y_vals[valid][.!ref_mask];
                            color = friction_colors[j],
                            markersize = 12,
                            marker = marker_shapes[j],
                            label = "ε̇: $(setting)"
                        )
                    end

                    if any(ref_mask)
                        scatter!(ax, x_vals[valid][ref_mask], y_vals[valid][ref_mask];
                            color = friction_colors[j],
                            markersize = 16,
                            marker = :star4,
                            strokecolor = :grey,
                            strokewidth = 2,
                            label = "reference model"
                        )
                    end
                end
            end
        end
    end




    # # Katmai: Roof ratio 2.0, Underpressure 166-205 MPa
    # # scatter!(ax, [2.0], [(166 + 205)/2], color = :red, markersize = 15, marker = :star5, label = "Katmai")
    # # rangebars!(ax, [2.0], [166.0], [205.0], color = :red)

    # # # Pinatubo: Roof ratio 2.4, Underpressure 265-312 MPa
    # # scatter!(ax, [2.4], [(265 + 312)/2], color = :blue, markersize = 15, marker = :star5, label = "Pinatubo")
    # # rangebars!(ax, [2.4], [265.0], [312.0], color = :blue)

    # # # # Pinatubo: Roof ratio 2.4, Underpressure 265-312 MPa
    # # scatter!(ax, [2.2], [4.5], color = :blue, markersize = 15, marker = :star5, label = "Pinatubo")
    # # rangebars!(ax, [2.2], [3.7], [5.3], color = :blue)
    # # rangebars!(ax, [4.5], [2.0], [2.4], direction = :x, color = :blue)

    # # # Crater Lake:  geshi 2013 roof ratio 1.0 depth (5-8km) bacon 1983, lipman 1997
    # scatter!(ax, [6.5], [55], color = :purple, markersize = 15, marker = :star5, label = "Crater Lake")
    # rangebars!(ax, [6.5], [50], [60], color = :purple)
    # rangebars!(ax, [55], [5.5], [7.5], direction = :x, color = :purple)

    # # krakatau: geshi 2013 roof ratio ? (deplus 1995m carey 1996)
    # scatter!(ax, [2.4], [12], color = :green, markersize = 15, marker = :star5, label = "Krakatau")
    # rangebars!(ax, [2.4], [10], [14], color = :green)
    # rangebars!(ax, [12], [2.2], [2.6], direction = :x, color = :green)

    # # Katmai: geshi 2013 roof ratio 2.4  depth 5-6km cioni 2008
    # scatter!(ax, [2.3], [12], color = :orange, markersize = 15, marker = :star5, label = "Katmai")
    # rangebars!(ax, [2.3], [10], [14], color = :orange)
    # rangebars!(ax, [12], [2.1], [2.5], direction = :x, color = :orange)

    # # Aniakchak: Bacon and others (2014), Miller and Smith 1998/1999
    # scatter!(ax, [7], [20], color = :brown, markersize = 15, marker = :star5, label = "Aniakchak")
    # rangebars!(ax, [7], [15], [25], color = :brown)
    # rangebars!(ax, [20], [6.5], [7.5], direction = :x, color = :brown)

    # # Kusdach Kamchatka Braitseva et al. (1996) 240AD Caldera
    # scatter!(ax, [4], [8], color = :pink, markersize = 15, marker = :star5, label = "Kusdach")
    # rangebars!(ax, [4], [7], [9], color = :pink)
    # rangebars!(ax, [8], [3], [5], direction = :x, color = :pink)

    # # tambora 1815 Foden (1986) after geshi 2013
    # scatter!(ax, [5], [33.2], color = :red, markersize = 15, marker = :star5, label = "Tambora")
    # rangebars!(ax, [5], [23.2], [43.2], color = :red)
    # rangebars!(ax, [33.2], [4.5], [5.5], direction = :x, color = :red)


    # # Vesuvuis: geshi 2013 roof ratio 2.4  depth 5-6km cioni 2008
    # scatter!(ax, [2.3], [1.4], color = :orange, markersize = 15, marker = :star5, label = "Vesuvius")
    # rangebars!(ax, [2.3], [1.2], [1.6], color = :orange)
    # rangebars!(ax, [1.4], [2.1], [2.5], direction = :x, color = :orange)


    # Geshi ? xlims!(ax, 1, 10) : xlims!(ax, 0.1, 3.0)
    # [:black, :blue, :green, :orange, :purple, :brown, :pink]
    xlims!(ax, 1e0, 1e2)
    ylims!(ax, 1e0, 1e9)
    # axislegend(ax, "Friction angle", position = :lb)

    fig[1, 2] = Legend(fig, ax, "Friction angle", position = :lt, merge = true, unique = true)
    # axislegend(ax, "Friction angle", position = :lt)
    display(fig)

    # save("./SmallScaleCaldera/Post_processing/Diameter_vs_Volume_fa_$(Int(k)).png", fig)
    # save("./SmallScaleCaldera/Post_processing/Diameter_vs_Volume_fa_$(Int(k)).svg", fig)
    # save("./SmallScaleCaldera/Post_processing/Diameter_vs_Volume_fa_$(Int(k)).pdf", fig)
end
