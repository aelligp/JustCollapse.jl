## Plotting routines for extracting underpressure at the onset of eruption

using JLD2, Statistics
using JustRelax, JustRelax.JustRelax2D

model = jldopen("/Users/pascalaellig/Documents/PhD/JustCollapse.jl/Systematics/Reference_models_local/ReRun_Caldera2D_2026-02-02_granite_d_5.0_r_1.75_ar_2.5_ex_0.0_phi_30.0/checkpoint/checkpoint0000.jld2")
# model = jldopen("/Volumes/Pascal/Caldera_Systematics/SmallCaldera_Systematics/No_extension/Caldera2D_2025-12-05_granite_d_5.0_r_1.75_ar_2.5_ex_0.0_phi_25.0/checkpoint/checkpoint0000.jld2")
# model = jldopen("/Users/pascalaellig/Documents/PhD/ScienceProjects/SmallScaleCollapse/Figures/04_mf_overpressure_checkpoint0000.jld2")

overpressure = model["overpressure"]
overpressure_04 = haskey(model, "overpressure_04") ? model["overpressure_04"] : nothing
overpressure_05 = haskey(model, "overpressure_05") ? model["overpressure_05"] : nothing
overpressure_time = model["overpressure_t"]
eruption_times = model["eruption_times"]
source_term = model["source_term"]
Q_strain_rate = haskey(model, "Q_strain_rate") ? model["Q_strain_rate"] : nothing
Q = haskey(model, "Q") ? model["Q"] : nothing

Volume = model["Volume"]
volume_times = model["volume_times"]
# Critical_timestep = 160
Critical_timestep = 145
overpressure_time[Critical_timestep]
overpressure[Critical_timestep]
Critical_undepressure_MPa = round(overpressure[Critical_timestep]/1e6;digits=3)

maximum_undepressure_MPa = round(minimum(overpressure)/1e6; digits=3)

n_missing = !isnothing(Q) ? length(overpressure_time) - length(Q) : 0
n_missing_Q_strain = !isnothing(Q_strain_rate) ? length(overpressure_time) - length(Q_strain_rate) : 0

# Append the last value of Q, n_missing times
if n_missing > 0
    append!(Q, fill(Q[end], n_missing))
end

if n_missing_Q_strain > 0
    append!(Q_strain_rate, fill(Q_strain_rate[end], n_missing_Q_strain))
end


eruption_times = round(last(model["eruption_times"]); digits=2)
eruption_volume = round(last(model["Volume"]) ./1e9; digits=2)
volume_times = round(model["volume_times"][Critical_timestep]; digits=3)


overpressure[Critical_timestep]
Critical_undepressure_MPa = round(overpressure[Critical_timestep]/1e6;digits=3)
eruption_times = model["eruption_times"]

mean_underpressure_MPa = round(mean(overpressure[(overpressure .< 0) .& (overpressure_time .> overpressure_time[150])])/1e6; digits=3)

using CairoMakie

moving_average(vs,n) = [sum(@view vs[i:(i+n-1)])/n for i in 1:(length(vs)-(n-1))]

let
# Create Figure and Axis
f = Figure(size = (800, 600))
ax = Axis(f[1, 1],
    xlabel = "Time [kyrs]",
    ylabel = "Dynamic Pressure [MPa]",
    # title = "Overpressure and Underpressure Evolution",
    yaxisposition = :left,
    yticklabelcolor = :black,
    ylabelsize = 24,
    xlabelsize = 24,
    yticklabelsize = 20,
    xticklabelsize = 20,
    yticks = -80:10:35
    )
ax1= Axis(f[1, 1],
    ylabel = "Injection and Eruption volume",
    yaxisposition = :right,
    yticklabelcolor = :purple,
    ylabelsize = 24,
    yticklabelsize = 20,
    xticklabelsvisible = false
    )

n_avg = 11
split_idx = Critical_timestep

# Part 1: Raw data up to split
t_raw = overpressure_time[1:split_idx]
p_raw = overpressure[1:split_idx] ./ 1e6

# Part 2: Smoothed data after split
t_smooth_input = overpressure_time[split_idx:end]
p_smooth_input = overpressure[split_idx:end] ./ 1e6

t_smooth = moving_average(t_smooth_input, n_avg)
p_smooth = moving_average(p_smooth_input, n_avg)

# Combine vectors
t_final = vcat(t_raw, t_smooth)
p_final = vcat(p_raw, p_smooth)




# 1. Plot the main line (Capture the object in ln1)
ln0 = lines!(ax , overpressure_time, overpressure./1e6, color=(:black, 0.3), linewidth = 3)
ln1 = lines!(ax, t_final, p_final, color=:black, label="Pressure evolution")
# ln1 = lines!(ax, moving_average(overpressure_time, 11), moving_average(overpressure./1e6, 11), color=:black, label="Pressure evolution")


# Plot the secondary line (Capture the object in ln2)
ln2 = lines!(ax1, overpressure_time, (Q_strain_rate./1e9), color=:purple, label="Injection and Eruption volume", linestyle = :dash)


# 2. Add the cross at the critical timestep (Capture the object in sc1)
sc1 = scatter!(ax, [overpressure_time[Critical_timestep]], [overpressure[Critical_timestep]]./1e6,
    # marker = :xcross,  # shape of the marker (can also use :x or :cross)
    marker = 'x',  # shape of the marker (can also use :x or :cross)
    color = :orange,
    markersize = 24,
    label = "Roof failure Figure 2d ($(Int64(round(Critical_undepressure_MPa;digits = 0))) MPa)"
)

sc2 = scatter!(ax, [overpressure_time[150]], [overpressure[150]]./1e6,
    # marker = :xcross,  # shape of the marker (can also use :x or :cross)
    marker = 'x',  # shape of the marker (can also use :x or :cross)
    color = :red,
    markersize = 24,
    label = "Figure 2b ($(Int64(round(overpressure[150]./1e6; digits = 0))) MPa)"
)
sc3 = scatter!(ax, [overpressure_time[155]], [overpressure[155]]./1e6,
    # marker = :xcross,  # shape of the marker (can also use :x or :cross)
    marker = 'x',  # shape of the marker (can also use :x or :cross)
    color = :purple,
    markersize = 24,
    label = "Figure 2c ($(Int64(round(overpressure[155]./1e6; digits = 0))) MPa)"
)

sc4 = scatter!(ax, [overpressure_time[810]], [overpressure[810]]./1e6,
    # marker = :xcross,  # shape of the marker (can also use :x or :cross)
    marker = 'x',  # shape of the marker (can also use :x or :cross)
    color = :brown,
    markersize = 24,
    label = "Figure 2e ($(Int64(round(overpressure[810]./1e6; digits = 0))) MPa)"
)
sc5 = scatter!(ax, [overpressure_time[1280]], [overpressure[1280]]./1e6,
    # marker = :xcross,  # shape of the marker (can also use :x or :cross)
    marker = 'x',  # shape of the marker (can also use :x or :cross)
    color = :green,
    markersize = 24,
    label = "End of simulation Figure 2f ($(Int64(round(overpressure[1280]./1e6; digits = 0))) MPa)"
)

hline1 = hlines!(ax, mean_underpressure_MPa; xmin = 0.4, xmax = 0.95, color=:red, linestyle=:dash, label="Mean underpressure $(Int64(round(mean_underpressure_MPa;digits = 0))) MPa")

# Create a combined legend on the primary axis (ax)
# axislegend(ax, [ln1, sc1, ln2, hline1], ["Over- & Underpressure", "Roof failure (P= $(Critical_undepressure_MPa) MPa)", "Injection and Eruption volume", "Mean underpressure $(mean_underpressure_MPa) MPa"], position = :rt)
# axislegend(ax, [ln1, ln2, sc1, sc2, sc3, sc4, hline1], ["Pressure evolution", "Injection and Eruption volume", "Roof failure (P= $(Critical_undepressure_MPa) MPa)", "First faults (P= $(Critical_undepressure_MPa) MPa)", "First faults (P= $(Critical_undepressure_MPa) MPa)", "Mean underpressure after roof failure \n (P = $(mean_underpressure_MPa)) MPa"], position = :rt)
axislegend(ax)#, ["Pressure evolution", "Injection and Eruption volume", "Roof failure (P= $(Critical_undepressure_MPa) MPa)", "First faults (P= $(round(overpressure[150]/1e6; digits=3)) MPa)", "First faults (P= $(round(overpressure[155]/1e6; digits=3)) MPa)", "First faults (P= $(round(overpressure[1280]/1e6; digits=3)) MPa)", "Mean underpressure after roof failure \n (P = $(mean_underpressure_MPa)) MPa"], position = :rt)
ylims!(ax, minimum(overpressure)./1e6 - 5, maximum(overpressure)./1e6 + 5)
ylims!(ax1, minimum(overpressure)./1e6 - 5, maximum(overpressure)./1e6 + 5)
display(f)
save("Overpressure_with_onset.png", f)
save("Overpressure_with_onset.svg", f)
save("Overpressure_with_onset.pdf", f)
end
