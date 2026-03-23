using JLD2, JustRelax, JustRelax.JustRelax2D
using CairoMakie

dir = "/Users/pascalaellig/Documents/PhD/JustCollapse.jl/Systematics/Reference_models_local/Eruptibility/plotting"
files = readdir(dir)

# reference_model = "d_5.0_r_1.75_ar_2.5_ex_0.0_phi_30.0"
# lower_end_member = "d_5.0_r_2.5_ar_2.0_ex_0.0_phi_30.0"
# upper_end_member = "d_5.0_r_1.75_ar_0.85_ex_0.0_phi_30.0"
models = ["d_5.0_r_1.75_ar_2.5_ex_0.0_phi_30.0", "d_5.0_r_2.5_ar_2.0_ex_0.0_phi_30.0", "d_5.0_r_1.75_ar_0.85_ex_0.0_phi_30.0"]

melt_fractions = [03, 04, 05]
let
fig = Figure(size = (1200, 800))

for (i_idx, i) in enumerate(models), (j_idx, j) in enumerate(melt_fractions)
    model_file = "Caldera2D_granite_mf_0$(j)_$(i)"


    model = jldopen(joinpath(dir, model_file, "checkpoint", "checkpoint0000.jld2"))
    overpressure_03 = model["overpressure"]
    overpressure_04 = haskey(model, "overpressure_04") ? model["overpressure_04"] : nothing
    overpressure_05 = haskey(model, "overpressure_05") ? model["overpressure_05"] : nothing
    overpressure_time = model["overpressure_t"]
    eruption_times = model["eruption_times"]
    source_term = haskey(model, "source_term") ? model["source_term"] : nothing
    Q_strain_rate = haskey(model, "Q_strain_rate") ? model["Q_strain_rate"] : nothing
    Q = haskey(model, "Q") ? model["Q"] : nothing
    Volume = haskey(model, "Volume") ? model["Volume"] : nothing
    volume_times = haskey(model, "volume_times") ? model["volume_times"] : nothing

    n_missing = !isnothing(Q) ? length(overpressure_time) - length(Q) : 0
    n_missing_Q_strain = !isnothing(Q_strain_rate) ? length(overpressure_time) - length(Q_strain_rate) : 0

    if n_missing > 0
    append!(Q, fill(Q[end], n_missing))
    end

    if n_missing_Q_strain > 0 || n_missing == nothing
        append!(Q_strain_rate, fill(Q_strain_rate[end], n_missing_Q_strain))
    end

    # Plotting logic for each model
    ax_i = Axis(fig[i_idx, j_idx],
        xlabel = (i_idx == 3 && j_idx == 2) ? "Time [kyrs]" : "",
        ylabel = (i_idx == 2 && j_idx == 1) ? "Dynamic Pressure [MPa]" : "",
        yaxisposition = :left,
        yticklabelcolor = :black,
        ylabelsize = 24,
        xlabelsize = 24,
        yticklabelsize = 20,
        xticklabelsize = 20,
    )

    ln0 = lines!(ax_i , overpressure_time, overpressure_03./1e6, color=(:black, j == 3 ? 1.0 : 0.3), linewidth = 3, label="ϕ = 30%")
    ln2 = !isnothing(overpressure_04) ? lines!(ax_i , overpressure_time, overpressure_04./1e6, color=(:blue, j == 4 ? 1.0 : 0.3), linewidth = 3, label = "ϕ = 40%") : nothing
    ln3 = !isnothing(overpressure_05) ? lines!(ax_i , overpressure_time, overpressure_05./1e6, color=(:green, j == 5 ? 1.0 : 0.3), linewidth = 3, label = "ϕ = 50%") : nothing
    xlims!(ax_i, 0, maximum(overpressure_time) + 15)

    panel_label = string(Char('a' + (i_idx - 1) * 3 + (j_idx - 1)))
    inset_ax = Axis(fig[i_idx, j_idx], width = Relative(0.07), height = Relative(0.12), halign = :left, valign = :top, backgroundcolor = :gray90)
    hidedecorations!(inset_ax); hidespines!(inset_ax)
    text!(inset_ax, 0.5, 0.5, text = panel_label, space = :relative, align = (:center, :center), fontsize = 18, color = :black)
    !isnothing(overpressure_05) ? println("Minimum and maximum overpressure for model $(i) and melt fraction $(j): \n ", minimum(overpressure_05)/1e6, " MPa, ", maximum(overpressure_03)/1e6, " MPa") : nothing
    (i_idx == 3 && j_idx == 3) ? fig[4, 1:3] = Legend(fig, [ln0, ln2, ln3], ["30% melt fraction", "40% melt fraction", "50% melt fraction"], orientation = :horizontal, fontsize = 24, "Eruption triggered at", titleposition = :top, titlesize = 24, labelsize = 20) : nothing
end
for (i, label) in enumerate(["Ref", "Lower end-member", "Upper end-member"])
    Box(fig[i, 4], color = :gray90)
    Label(fig[i, 4], label, rotation = pi/2, tellheight = false, fontsize = 18, color = :black)
end
for (j, label) in enumerate([
    rich("ϕ", subscript("m"), " = 30%"),
    rich("ϕ", subscript("m"), " = 40%"),
    rich("ϕ", subscript("m"), " = 50%"),
])
    Box(fig[0, j], color = :gray90)
    Label(fig[0, j], label, tellwidth = false, fontsize = 18, color = :black)
end

display(fig)
save("./SmallScaleCaldera/Post_processing/Eruptibility_testing.png", fig)
save("./SmallScaleCaldera/Post_processing/Eruptibility_testing.pdf", fig)
save("./SmallScaleCaldera/Post_processing/Eruptibility_testing.svg", fig)
end
