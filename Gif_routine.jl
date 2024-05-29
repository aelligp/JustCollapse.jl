# Gif - routine
# Attention: plots needs %06d.png format
using Plots, Printf, FileIO

figdir = "./fig2D/GIF_Lugano/"
gifname = "GIF_Lugano_Plots";
nt = 43  #define


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

gif(anim, "$gifname"*".gif", fps = 5)
