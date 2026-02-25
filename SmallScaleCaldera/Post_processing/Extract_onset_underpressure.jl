using JLD2
using JustRelax, JustRelax.JustRelax2D


# Load model & read out the data
# model = jldopen("/Volumes/G-RAID MIRROR/Systematics_December/daint_systematics_december_no_ext/Caldera2D_2025-12-05_granite_d_5.0_r_2.25_ar_2.5_ex_0.0_phi_30.0/checkpoint/checkpoint0000.jld2")
# model = jldopen("/Volumes/Pascal/Caldera_Systematics/Systematics_December/Compression/Caldera2D_2025-12-10_granite_d_5.0_r_1.75_ar_1.5_ex_-1.0e-15_phi_25.0/checkpoint/checkpoint0000.jld2")
# model = jldopen("/Users/pascalaellig/Documents/PhD/JustCollapse.jl/Additional_Runs/Caldera2D_2025-12-17_granite_d_5.0_r_2.5_ar_3.0_ex_1.0e-15_phi_30.0/checkpoint/checkpoint0000.jld2")
# model = jldopen("/Volumes/Pascal/Caldera_Systematics/SmallCaldera_Systematics/Extension/Extension_Caldera2D_2025-12-10_granite_d_5.0_r_2.5_ar_1.5_ex_1.0e-15_phi_15.0/checkpoint/checkpoint0000.jld2")
# model = jldopen("/Volumes/Pascal/Caldera_Systematics/SmallCaldera_Systematics/No_extension/Caldera2D_2025-12-05_granite_d_5.0_r_1.75_ar_2.5_ex_0.0_phi_30.0/checkpoint/checkpoint0000.jld2")
# model = jldopen("/Users/pascalaellig/Desktop/084_systematics_checkpoint/checkpoint0000.jld2")
# model = jldopen("/Users/pascalaellig/Desktop/Volcano2D/Convergence_Test/Test_checkpoint0000.jld2")
model = jldopen("/Volumes/Pascal/Caldera_Systematics/SmallCaldera_Systematics/Reference_Case_Additional_Runs/ReRun_Caldera2D_2026-02-02_granite_d_5.0_r_1.75_ar_2.5_ex_0.0_phi_30.0/checkpoint/checkpoint0000.jld2")
overpressure = model["overpressure"]
overpressure_time = model["overpressure_t"]
eruption_times = model["eruption_times"]
Volume = model["Volume"]
volume_times = model["volume_times"]
Critical_timestep = 130
overpressure_time[Critical_timestep]
overpressure[Critical_timestep]
Critical_undepressure_MPa = round(overpressure[Critical_timestep]/1e6;digits=3)


maximum_undepressure_MPa = round(minimum(overpressure)/1e6; digits=3)



eruption_times = round(last(model["eruption_times"]); digits=3)
eruption_volume = round(last(model["Volume"]) ./1e9; digits=3)
volume_times = round(model["volume_times"][Critical_timestep]; digits=3)



### Extract model parameters and write to file

using XLSX
using JLD2
using DataFrames
data = DataFrame(XLSX.readtable("/Users/pascalaellig/Documents/PhD/JustCollapse.jl/Onset_of_caldera_collapse_CSV.xlsx", "Failed_Unused models"))
extractions = true

for extract in extractions
name = data[:, 1]
bg_strain = data[:, 2]
# diameter = data[:, 3]
Topography = data[:, 4]
depth = data[:, 5]
radius = data[:, 6]
aspect_ratio = data[:, 7]
friction_angle = data[:, 9]
timestep = data[:, 12]



for model_name in name
    # model = jldopen("/Volumes/Pascal/Caldera_Systematics/SmallCaldera_Systematics/Reference_Case_Additional_Runs/$(model_name)/checkpoint/checkpoint0000.jld2")
    model = try
        model = jldopen("/Volumes/Pascal/Caldera_Systematics/SmallCaldera_Systematics/Additional_runs/$(model_name)/checkpoint/checkpoint0000.jld2")
    catch e
        println("Could not open model: $(model_name), skipping...")
        continue
    end
    # model = jldopen("/Volumes/Pascal/Caldera_Systematics/SmallCaldera_Systematics/Reference_Case_Additional_Runs/$(model_name)/checkpoint/checkpoint0000.jld2")

    overpressure = model["overpressure"]
    overpressure_time = model["overpressure_t"]
    Critical_timestep = timestep[findfirst(==(model_name), name)]
    if Critical_timestep === missing
        println("No timestep found for model: $(model_name), skipping...")
        continue
    end
    Critical_undepressure_MPa = round(overpressure[Critical_timestep]/1e6;digits=3)
    maximum_undepressure_MPa = round(minimum(overpressure)/1e6; digits=3)
    eruption_times = round(last(model["eruption_times"]); digits=3)
    eruption_volume = round(last(model["Volume"]) ./1e9; digits=3)
    volume_times = round(model["volume_times"][Critical_timestep]; digits=3)

    # now write to file and specific column
    data[findfirst(==(model_name), name), 16] = Critical_undepressure_MPa
    data[findfirst(==(model_name), name), 17] = maximum_undepressure_MPa
    data[findfirst(==(model_name), name), 10] = eruption_times
    data[findfirst(==(model_name), name), 11] = eruption_volume
    data[findfirst(==(model_name), name), 12] = volume_times
end

# Write the updated data back to the Excel file
XLSX.writetable("/Users/pascalaellig/Documents/PhD/JustCollapse.jl/Onset_of_caldera_collapse_CSV_copy.xlsx",
                data,
                overwrite=true,
                sheetname="Failed_Unused models")
end



using XLSX
using DataFrames
data = DataFrame(XLSX.readtable("/Users/pascalaellig/Documents/PhD/JustCollapse.jl/Onset_of_caldera_collapse_CSV.xlsx", "Systematics"))
extractions = true


for extract in extractions
name = data[25:28, 1]
bg_strain = data[:, 2]
# diameter = data[:, 3]
Topography = data[:, 4]
depth = data[:, 5]
radius = data[:, 6]
aspect_ratio = data[:, 7]
friction_angle = data[:, 9]
timestep = data[:, 15]
friction_angle = data[:, 9]

# reference_model = append!(String[], "Caldera2D_2025-12-05_granite_d_5.0_r_1.75_ar_2.5_ex_0.0_phi_$(friction_angle[25:28])")


for model_name in name
    # model = jldopen("/Volumes/Pascal/Caldera_Systematics/SmallCaldera_Systematics/Reference_Case_Additional_Runs/$(model_name)/checkpoint/checkpoint0000.jld2")
    model = try
        model = jldopen("/Volumes/Pascal/Caldera_Systematics/SmallCaldera_Systematics/Additional_Runs/$(model_name)/checkpoint/checkpoint0000.jld2")
    catch e
        println("Could not open model: $(model_name), skipping...")
        continue
    end
    # model = jldopen("/Volumes/Pascal/Caldera_Systematics/SmallCaldera_Systematics/Reference_Case_Additional_Runs/$(model_name)/checkpoint/checkpoint0000.jld2")

    overpressure = model["overpressure"]
    overpressure_time = model["overpressure_t"]
    Critical_timestep = timestep[findfirst(==(model_name), name)]
    if Critical_timestep === missing
        println("No timestep found for model: $(model_name), skipping...")
        continue
    end
    Critical_undepressure_MPa = round(overpressure[Critical_timestep]/1e6;digits=3)
    maximum_undepressure_MPa = round(minimum(overpressure)/1e6; digits=3)
    eruption_times = round(last(model["eruption_times"]); digits=3)
    eruption_volume = round(last(model["Volume"]) ./1e9; digits=3)
    volume_times = round(model["volume_times"][Critical_timestep]; digits=3)

    # now write to file and specific column
    data[findfirst(==(model_name), name), 16] = Critical_undepressure_MPa
    data[findfirst(==(model_name), name), 17] = maximum_undepressure_MPa
    data[findfirst(==(model_name), name), 10] = eruption_times
    data[findfirst(==(model_name), name), 11] = eruption_volume
    data[findfirst(==(model_name), name), 12] = volume_times
end

# Write the updated data back to the Excel file
XLSX.writetable("/Users/pascalaellig/Documents/PhD/JustCollapse.jl/Onset_of_caldera_collapse_CSV_copy.xlsx",
                data,
                overwrite=true,
                sheetname="Reference_run_variations")
end
