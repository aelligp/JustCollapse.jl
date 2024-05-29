using GeoParams

CharDim     = GEO_units(length=40km, viscosity=1e20Pa*s);

Depth       = Array(GeoUnit(0km):GeoUnit(1km):GeoUnit(10km));
Depth_nondim= nondimensionalize(Depth,CharDim);
# Geotherm    = nondimensionalize(30K, CharDim)/nondimensionalize(1km, CharDim)
Geotherm    = nondimensionalize(GeoUnit(30K/1km),CharDim)
# Geotherm_C  = nondimensionalize(30C, CharDim)/nondimensionalize(1km, CharDim)
Geotherm_C = nondimensionalize(GeoUnit(30C),CharDim)/nondimensionalize(GeoUnit(1km),CharDim)
Gradient    = nondimensionalize(273.15K,CharDim) .+ Geotherm * Depth_nondim;
Temp_K_dim  = dimensionalize(Gradient, K, CharDim)

Gradient_C  = nondimensionalize(0C,CharDim) .+ Geotherm_C * Depth_nondim;
Temp_C_dim  = dimensionalize(Gradient_C, C, CharDim)


geot = GeoUnit(30C/1km)#/GeoUnits(1km)