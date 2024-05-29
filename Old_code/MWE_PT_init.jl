
function init_topography(Phase, temperature)
    ni, nj = size(Phase)
    for i in 1:ni
        for j in 1:nj
            if Phase[i, j] == 3
                println("i: $i, j: $j") 
                temperature[i,j] = nondimensionalize(0.0C, CharDim)
                break
            end
        end
    end

end

function init_temperature!(temperature, z, lz, phases, geotherm, CharDim::GeoUnits) 
    ni, nj = size(phases) 

    for j in 1:nj, i in 1:ni
        depth = z[j]
        if phases[i,j] == 3.0
            temperature[i,j] = nondimensionalize(0.0C,CharDim)
        elseif phases[i,j] == 1.0 
            if depth > 0.0    
                temperature[i,j] = abs(geotherm * abs(lz-(depth+minimum(depth))))  
                
            else         
                temperature[i,j] = (abs(geotherm * abs(lz - depth)))
            end
        end
    end
end
init_temperature!(thermal.T, xvi[2], li[2], Phase, GeoT, CharDim)

function init_pressure!(pressure, z, lz, phases, ρg, CharDim::GeoUnits) 
    ni, nj = size(phases) 

    for j in 1:nj, i in 1:ni
        depth = z[j]
        if phases[i,j] == 3.0
            pressure[i,j] = nondimensionalize(0.0C,CharDim)
        elseif phases[i,j] == 1.0 
            if depth > 0.0    
                pressure[i,j] = abs(ρg * abs(lz-(depth+minimum(depth))))  
                
            else         
                pressure[i,j] = (abs(ρg * abs(lz - depth)))
            end
        end
    end
end
init_pressure!(stokes.P, xvi[2], li[2], Phase, ρg, CharDim)
# new_depth = 
# lx, lz                  = (Grid2D.L[1]), (Grid2D.L[2]) # nondim if CharDim=CharDim
# li                      = lx, lz
# b_width                 = (4, 4, 0) #boundary width
# origin                  = Grid2D.min[1], -li[2]
# igg                     = igg
# di                      = @. li / (nx_g(), ny_g()) # grid step in x- and y-direction
# pT_init_grid_center, pT_init_grid_vertex = lazy_grid(di, li, ni; origin=(Grid2D.min[1], -li[2]))
# topography = [y for x in pT_init_grid_vertex[1], y in pT_init_grid_vertex[2]]    
# function init_temperature!(temperature, z, phases, geotherm, CharDim::GeoUnits)
#     ni, nj = size(phases)
#     for j in 1:nj
#         found_1 = false
#         for i in 1:ni
#             if phases[i, j] == 1.0
#                 found_1 = true
#                 depth = -z[j]
#                 if depth < 0.0
#                     # temperature[i:end, j] .= nondimensionalize(273.0C, CharDim)
#                     dTdZ = geotherm / nj
#                     offset = minimum(temperature)
#                     temperature[i:end, j] .= (depth - depth[i]) * dTdZ + offset
#                 end
#             elseif phases[i, j] == 3.0
#                 temperature[i, j] = nondimensionalize(0.0C, CharDim)
#             end
#         end
#     end
# end


topography = [y for x in xvi[1], y in xvi[2]]


# @parallel_indices (i, j) function init_T!(T, z)
#     depth = -z[j]

#     # (depth) because we have 15km of sticky air
#     if depth < 0e0
#         T[i + 1, j] = 273e0

#     elseif 0e0 ≤ (depth) < 35e3
#         dTdZ        = (923-273)/35e3
#         offset      = 273e0
#         T[i + 1, j] = (depth) * dTdZ + offset
    
#     elseif 110e3 > (depth) ≥ 35e3
#         dTdZ        = (1492-923)/75e3
#         offset      = 923
#         T[i + 1, j] = (depth - 35e3) * dTdZ + offset

#     elseif (depth) ≥ 110e3 
#         dTdZ        = (1837 - 1492)/590e3
#         offset      = 1492e0
#         T[i + 1, j] = (depth - 110e3) * dTdZ + offset

#     end
    
#     return nothing
# end


xs = 1:0.2:5
A = log.(xs)

using Interpolations
evo_t = Float64[]
phases_new = Int64.(zeros(nx+1,ny+1))
depth_corrected = zeros(nx+1,ny+1)
depth3D_corrected = zeros(nx+1,ny+1,nz+1)


topo_interp = linear_interpolation(xvi[1], Topo_nondim)
function update_depth!(depth_corrected,topo_interp, x, y)
    nx, ny = length(x), length(y)
  
    for i in 1:nx, j in 1:ny
        
        # vertex coordinates
        xv, yv = x[i], y[j]
        # topography at vertex
        y_topo = topo_interp(xv)
        # depth
        depth = yv

        depth_corrected[i,j] = abs(depth - y_topo)
    end
    return nothing
end

update_depth!(depth_corrected,topo_interp, xvi...)

@parallel_indices function update_phases_topo(phases, depth_corrected)
    
end


### This function doesn not work !!! interpolations cannot be done on GPU (yet)
# topo_interp = linear_interpolation(xvi[1], Topo_nondim)
# @parallel_indices (i,j) function update_depth!(depth_corrected,topo_interp, x, y)

      
#     # vertex coordinates
#     xv, yv = x[i], y[j]
#     # topography at vertex
#     y_topo = topo_interp(xv)
#     # depth
#     depth = yv

#     depth_corrected[i,j] = abs(depth - y_topo)

#     return nothing
# end

# @parallel (@idx ni) update_depth!(depth_corrected,topo_interp, xvi...)

# interpolant object
topo_interp = linear_interpolation((topo_x, topo_y),topo_data)

function update_depth3D!(depth3D_corrected,topo_interp, x, y, z)
    nx, ny,nz = length(x), length(y), length(z)

  
    for i in 1:nx, j in 1:ny, k in 1:nz
        # vertex coordinates
        xv, yv, zv = x[i], y[j], z[k]
        # topography at vertex
        z_topo = topo_interp(xv,yv)
        # depth
        depth = zv

        depth3D_corrected[i,j,k] = abs(depth - z_topo)
    end
end

update_depth3D!(depth3D_corrected,Topo_2D_test, xvi3D...)

# xvi_corrected = [y for x in x_corrected, y in y_corrected]

function init_phases_topo!(phases_new, depth_corrected, topo_y, z)
    nx, ny = length(z[1]), length(z[2])
    topo_x = [x for x in z[1]]
  
    for i in 1:nx, j in 1:ny
        # interpolant object
        topo_interp = linear_interpolation(topo_x, topo_y)
        # vertex coordinates
        xv, yv = z[1][i], z[2][j]
        # topography at vertex
        y_topo = topo_interp(xv)
        # maximum value the y-coordinate
        z0 = maximum(z[2])
        # depth
        depth = yv

        depth_corrected[i,j] = abs(depth - y_topo)
        if depth > y_topo
            phases_new[i,j] = Int64(3)
        else 
            phases_new[i,j] = Int64(1)
        end
    end
end

init_phases_topo!(phases_new, depth_corrected, Topo_nondim,xvi)

function init_temperature!(temperature, topo_y, z, lz, depth_corrected,phases, phases_new, geotherm)#, CharDim::GeoUnits) 
    nx, ny = size(phases) 
    topo_x = [x for x in z[1]]
  
    for i in 1:nx, j in 1:ny
        # interpolant object
        topo_interp = linear_interpolation(topo_x, topo_y)
        # vertex coordinates
        xv, yv = z[1][i], z[2][j]
        # topography at vertex
        y_topo = topo_interp(xv)
        # maximum value the y-coordinate
        z0 = maximum(z[2])
        # depth
        depth = yv

        # correct depth
        # if depth > y_topo
        # if depth > 0.0
            depth_corrected[i,j] = abs(depth - y_topo)
            if depth > y_topo
                phases_new[i,j] = Int64(3)
            else 
                phases_new[i,j] = Int64(1)
            end

#             # depth_corrected[phases_new .== 3] .= nondimensionalize(0.0km,CharDim)
#             # depth_corrected[phases_new .== 3] .= 0.0
#         # elseif depth < y_topo
#         #     depth_corrected = abs(depth) + y_topo
#         #     phases_new[i,j] = Int64(1)
#         # end
#         # if phases_new[i,j] == 3
#             # temperature[i+1,j] = nondimensionalize(0.0C,CharDim)
#         # elseif phases_new[i,j] == 1 
#             # temperature[i+1,j] = (geotherm * (depth_corrected))
#             # temperature[i+1,j] = (nondimensionalize(30C,CharDim) * (depth_corrected[i,j]))
#             # temperature[i+1,j] = depth_corrected[i,j] * geotherm
#             # temperature[i+1,j] = (ustrip.(dimensionalize(depth_corrected[i,j],km,CharDim)) .* 30)
        
    end
    
# #     # return phases_new
end
init_temperature!(thermal.T, Topo_nondim, xvi, li[2], depth_corrected, Phase, phases_new, geotherm)#, CharDim)
# nondimensionalize(thermal.T,CharDim)

# thermal.T = nondimensionalize(thermal.T,CharDim)

# thermal.T .= abs.(thermal.T) .+ maximum(thermal.T)




# topo_y =
# thermal.T = abs.(thermal.T)

            # temperature[i+1,j] = abs(temperature[i+1,j])
            # temperature[i+1,j] = abs(nondimensionalize(-30C,CharDim) * (depth_corrected)) #+ nondimensionalize(273C,CharDim)
            # temperature[i+1,j] = (nondimensionalize(-30C,CharDim) * (depth_corrected))
            # if depth > 0.0    
            #     temperature[j] = (geotherm * abs((depth_corrected)))  
            #     # temperature[i,j] = abs(geotherm * (depth_corrected))
                
            # else         
            #     # temperature[i,j] = (abs(geotherm * abs(lz - depth)))
            #     temperature[i,j] = (geotherm * abs(depth_corrected))
            # end

# Grid3D                  = CreateCartGrid(size=(nx,ny),x=((Topo.x.val[1,1,1])km,(Topo.x.val[end,1,1])km), y=((Topo.y.val[1,1,1])km,(Topo.y.val[1,end,1])km),z=(minimum(Topo_Cart.z.val)km,maximum(Topo_Cart.z.val)km))#, CharDim=CharDim);  
# X,Y,Z                   = XYZGrid(Grid3D.coord1D...);
# Topo_3D_Cart            = CartData(X,Y,Z,(Depth=Z,));
# Topo_3D_drop            = dropdims(Topo_3D_Cart, dims=3)    

# # Topo_2D_nondim          = nondimensionalize(Topo_3D_Cart, CharDim)
# lx, ly, lz                  = (Grid3D.L[1]), (Grid3D.L[2]), (Grid3D.L[3]) # nondim if CharDim=CharDim
# li                      = lx, ly, lz
# b_width                 = (4, 4, 1) #boundary width
# origin                  = Grid3D.min[1], Grid3D.min[2], Grid3D.min[3]
# igg                     = igg
# di                      = @. li / (nx_g(), ny_g(), ny_g()) # grid step in x- and y-direction
# xci3D, xvi3D                = lazy_grid(di, li, ni; origin=origin) #non-dim nodes at the center and the vertices of the cell (staggered grid)

# Topo_2D_nondim