# Gif - routine
using Plots, Printf, FileIO

figdir = "./fig2D/shear_test"
gifname = "dike_toy_model";
nt = 25  #define


file_names = String[]
# Rename the files using sprintf
# for i in 1:nt
#     old_file_name = @sprintf("1%04d.png", i)
#     new_file_name = @sprintf("%06d.png", i)
#     FileIO.rename(old_file_name, new_file_name)
# end
for i = 1:nt
    push!(file_names, @sprintf("%06d.png", i))
    # save(file_names, @sprintf("%06d.png", i))
end
anim = Animation(figdir, file_names)

gif(anim, "$gifname"*".gif", fps = 15)
