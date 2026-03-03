## Plotting routines for extracting underpressure at the onset of eruption

using JLD2, Statistics
using JustRelax, JustRelax.JustRelax2D

model = jldopen("/Users/pascalaellig/Documents/PhD/JustCollapse.jl/Systematics/Reference_models_local/ReRun_Caldera2D_2026-02-02_granite_d_5.0_r_1.75_ar_2.5_ex_0.0_phi_30.0/checkpoint/checkpoint0000.jld2")
# model = jldopen("/Volumes/Pascal/Caldera_Systematics/SmallCaldera_Systematics/No_extension/Caldera2D_2025-12-05_granite_d_5.0_r_1.75_ar_2.5_ex_0.0_phi_25.0/checkpoint/checkpoint0000.jld2")
model = jldopen("/Users/pascalaellig/Documents/PhD/ScienceProjects/SmallScaleCollapse/Figures/04_mf_overpressure_checkpoint0000.jld2")

overpressure = model["overpressure"]
overpressure_04 = model["overpressure_04"]
overpressure_05 = model["overpressure_05"]
overpressure_time = model["overpressure_t"]
eruption_times = model["eruption_times"]
source_term = model["source_term"]
Q_strain_rate = model["Q_strain_rate"]
Q = model["Q"]

Volume = model["Volume"]
volume_times = model["volume_times"]
# Critical_timestep = 160
Critical_timestep = 145
overpressure_time[Critical_timestep]
overpressure[Critical_timestep]
Critical_undepressure_MPa = round(overpressure[Critical_timestep]/1e6;digits=3)

maximum_undepressure_MPa = round(minimum(overpressure)/1e6; digits=3)

n_missing = length(overpressure_time) - length(Q)
n_missing_Q_strain = length(overpressure_time) - length(Q_strain_rate)

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
ln2 = lines!(ax1, overpressure_time, (Q./1e9), color=:purple, label="Injection and Eruption volume", linestyle = :dash)


# 2. Add the cross at the critical timestep (Capture the object in sc1)
sc1 = scatter!(ax, [overpressure_time[Critical_timestep]], [overpressure[Critical_timestep]]./1e6,
    # marker = :xcross,  # shape of the marker (can also use :x or :cross)
    marker = 'x',  # shape of the marker (can also use :x or :cross)
    color = :orange,
    markersize = 24,
    label = "Roof failure Figure 2d ($(round(Critical_undepressure_MPa;digits =2)) MPa)"
)

sc2 = scatter!(ax, [overpressure_time[150]], [overpressure[150]]./1e6,
    # marker = :xcross,  # shape of the marker (can also use :x or :cross)
    marker = 'x',  # shape of the marker (can also use :x or :cross)
    color = :red,
    markersize = 24,
    label = "Figure 2b ($(round(overpressure[150]./1e6; digits=2)) MPa)"
)
sc3 = scatter!(ax, [overpressure_time[155]], [overpressure[155]]./1e6,
    # marker = :xcross,  # shape of the marker (can also use :x or :cross)
    marker = 'x',  # shape of the marker (can also use :x or :cross)
    color = :purple,
    markersize = 24,
    label = "Figure 2c ($(round(overpressure[155]./1e6; digits=2)) MPa)"
)

sc4 = scatter!(ax, [overpressure_time[810]], [overpressure[810]]./1e6,
    # marker = :xcross,  # shape of the marker (can also use :x or :cross)
    marker = 'x',  # shape of the marker (can also use :x or :cross)
    color = :brown,
    markersize = 24,
    label = "Figure 2e ($(round(overpressure[810]./1e6; digits = 2)) MPa)"
)
sc5 = scatter!(ax, [overpressure_time[1280]], [overpressure[1280]]./1e6,
    # marker = :xcross,  # shape of the marker (can also use :x or :cross)
    marker = 'x',  # shape of the marker (can also use :x or :cross)
    color = :green,
    markersize = 24,
    label = "End of simulation Figure 2f ($(round(overpressure[1280]./1e6; digits = 2)) MPa)"
)

hline1 = hlines!(ax, mean_underpressure_MPa; xmin = 0.4, xmax = 0.95, color=:red, linestyle=:dash, label="Mean underpressure $(round(mean_underpressure_MPa;digits=2)) MPa")

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

# Eruptibilty plot of different simulations
let

# Create Figure and Axis
f = Figure(size = (1200, 1000))
ax = Axis(f[1, 1],
    aspect = DataAspect(),
    xlabel = "Time [kyrs]",
    ylabel = "Dynamic Pressure [MPa]",
    title = "Reference model eruption triggered at 40%",
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
    aspect = DataAspect(),
    ylabel = "Injection and Eruption volume",
    yaxisposition = :right,
    yticklabelcolor = :purple,
    ylabelsize = 24,
    yticklabelsize = 20,
    xticklabelsvisible = false
    )

ax2 = Axis(f[1, 2],
    aspect = DataAspect(),
    xlabel = "Time [kyrs]",
    ylabel = "Dynamic Pressure [MPa]",
    title = "Reference model eruption triggered at 50%",
    # title = "Overpressure and Underpressure Evolution",
    yaxisposition = :left,
    yticklabelcolor = :black,
    ylabelsize = 24,
    xlabelsize = 24,
    yticklabelsize = 20,
    xticklabelsize = 20,
    yticks = -80:10:35
    )

ax3= Axis(f[1, 2],
    aspect = DataAspect(),
    ylabel = "Injection and Eruption volume",
    yaxisposition = :right,
    yticklabelcolor = :purple,
    ylabelsize = 24,
    yticklabelsize = 20,
    xticklabelsvisible = false
    )

ax4 = Axis(f[2, 1],
    aspect = DataAspect(),
    xlabel = "Time [kyrs]",
    ylabel = "Dynamic Pressure [MPa]",
    title = "Roof ratio 2.0 evolution",
    yaxisposition = :left,
    yticklabelcolor = :black,
    ylabelsize = 24,
    xlabelsize = 24,
    yticklabelsize = 20,
    xticklabelsize = 20,
    yticks = -80:10:35
    )

ax5= Axis(f[2, 1],
    aspect = DataAspect(),
    ylabel = "Injection and Eruption volume",
    yaxisposition = :right,
    yticklabelcolor = :purple,
    ylabelsize = 24,
    yticklabelsize = 20,
    xticklabelsvisible = false
    )

ax6 = Axis(f[2, 2],
    aspect = DataAspect(),
    xlabel = "Time [kyrs]",
    ylabel = "Dynamic Pressure [MPa]",
    title = "Roof ratio 0.5 evolution",
    yaxisposition = :left,
    yticklabelcolor = :black,
    ylabelsize = 24,
    xlabelsize = 24,
    yticklabelsize = 20,
    xticklabelsize = 20,
    yticks = -80:10:35
    )

ax7= Axis(f[2, 2],
    aspect = DataAspect(),
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
ln1 = lines!(ax, t_final, p_final, color=:black, label="30% melt fraction")
ln2 = lines!(ax , overpressure_time, overpressure_04./1e6, color=(:blue, 0.3), linewidth = 3, label = "40% melt fraction")
ln3 = lines!(ax , overpressure_time, overpressure_05./1e6, color=(:green, 0.3), linewidth = 3, label = "50% melt fraction")
ln4 = lines!(ax1, overpressure_time, (Q_strain_rate./1e9), color=:purple, label="Injection and Eruption volume", linestyle = :dash)

ax2ln0 = lines!(ax2 , overpressure_time, overpressure./1e6, color=(:black, 0.3), linewidth = 3)
ax2ln1 = lines!(ax2, t_final, p_final, color=:black, label="Pressure evolution at 30% melt fraction")
ax2ln2 = lines!(ax2 , overpressure_time, overpressure_04./1e6, color=(:blue, 0.3), linewidth = 3, label = "40% melt fraction")
ax2ln3 = lines!(ax2 , overpressure_time, overpressure_05./1e6, color=(:green, 0.3), linewidth = 3, label = "50% melt fraction")
ax2ln4 = lines!(ax3, overpressure_time, (Q_strain_rate./1e9), color=:purple, label="Injection and Eruption volume", linestyle = :dash)

ax4ln0 = lines!(ax4 , overpressure_time, overpressure./1e6, color=(:black, 0.3), linewidth = 3)
ax4ln1 = lines!(ax4, t_final, p_final, color=:black, label="30% melt fraction")
ax4ln2 = lines!(ax4 , overpressure_time, overpressure_04./1e6, color=(:blue, 0.3), linewidth = 3, label = "40% melt fraction")
ax4ln3 = lines!(ax4 , overpressure_time, overpressure_05./1e6, color=(:green, 0.3), linewidth = 3, label = "50% melt fraction")
ax4ln4 = lines!(ax5, overpressure_time, (Q_strain_rate./1e9), color=:purple, label="Injection and Eruption volume", linestyle = :dash)

ax6ln0 = lines!(ax6 , overpressure_time, overpressure./1e6, color=(:black, 0.3), linewidth = 3)
ax6ln1 = lines!(ax6, t_final, p_final, color=:black, label="Pressure evolution at 30% melt fraction")
ax6ln2 = lines!(ax6 , overpressure_time, overpressure_04./1e6, color=(:blue, 0.3), linewidth = 3, label = "40% melt fraction")
ax6ln3 = lines!(ax6 , overpressure_time, overpressure_05./1e6, color=(:green, 0.3), linewidth = 3, label = "50% melt fraction")
ax6ln4 = lines!(ax7, overpressure_time, (Q_strain_rate./1e9), color=:purple, label="Injection and Eruption volume", linestyle = :dash)
# ln1 = lines!(ax, moving_average(overpressure_time, 11), moving_average(overpressure./1e6, 11), color=:black, label="Pressure evolution")
linkaxes!(ax, ax3, ax4, ax6)

axislegend(ax, position = :lb)
ylims!(ax, minimum(overpressure)./1e6 - 5, maximum(overpressure)./1e6 + 5)
ylims!(ax1, minimum(overpressure)./1e6 - 5, maximum(overpressure)./1e6 + 5)
ylims!(ax2, minimum(overpressure)./1e6 - 5, maximum(overpressure)./1e6 + 5)
ylims!(ax3, minimum(overpressure)./1e6 - 5, maximum(overpressure)./1e6 + 5)
ylims!(ax4, minimum(overpressure)./1e6 - 5, maximum(overpressure)./1e6 + 5)
ylims!(ax5, minimum(overpressure)./1e6 - 5, maximum(overpressure)./1e6 + 5)
ylims!(ax6, minimum(overpressure)./1e6 - 5, maximum(overpressure)./1e6 + 5)
ylims!(ax7, minimum(overpressure)./1e6 - 5, maximum(overpressure)./1e6 + 5)
display(f)

save("./SmallScaleCaldera/Post_processing/Eruptibility_testing_overpressure.png", f)
save("./SmallScaleCaldera/Post_processing/Eruptibility_testing_overpressure.svg", f)
save("./SmallScaleCaldera/Post_processing/Eruptibility_testing_overpressure.pdf", f)
end
