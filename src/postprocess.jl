using HDF5, Measurements, ArgParse, FourierFlows, CUDA
include("utils.jl")

# Parse argument
aps = ArgParseSettings()
@add_arg_table! aps begin
	"--src"
	help = "Path of the source file."
	arg_type = String
	default = dirname(@__DIR__)
	"--des"
	help = "Path of the destination file."
	arg_type = String
	default = dirname(@__DIR__)
	"--ngrid", "-n"
	help = "Resolution of the source data."
	arg_type = Int
	default = 128
	"--gpu"
	help = "Using GPU acceleration."
	action = :store_true
end
args = parse_args(aps)

# Global Constants
ngrid = args["ngrid"]
srcPath = args["src"]*"/$(ngrid)"
desPath = args["des"]
dev = args["gpu"] ? GPU() : CPU()
desFile = h5open(desPath * "/postdata.h5", "w")
Reynolds = map(readdir(srcPath)) do str
	m = match(r"(\d+\.\d+)\.h5", str)
	parse(Float64, m[1])
end
srcFiles = Dict([Re=>h5open(srcPath * "/$(Re).h5", "r") for Re in Reynolds])
grid = TwoDGrid(dev; nx = ngrid, Lx = 2π)

# Inteface of the source Data
lpFilter(Re) = device_array(dev)(float(grid.Krsq .< (ngrid / 3sqrt(Re) - 5)^2))

function getζhs(Re, isfilter)
	ζhs = Iterators.map(srcFiles[Re]) do ds
		ζh = read(ds)
		device_array(dev)(ζh)
	end
	if isfilter
		fh = lpFilter(Re)
		return Iterators.map(ζh -> ζh .* fh, ζhs)
	else
		return ζhs
	end
end

function statOver(func, ζhs)
	ndata = length(ζhs)
	mean = sum(func, ζhs) ./ ndata
	var = sum(ζhs) do ζh
		(func(ζh) .- mean) .^ 2
	end ./ ndata

	return (mean, var)
end

# Analysed data
function energySpectrumStat(Re)
	ζhs = getζhs(Re, false)
	des = create_group(desFile, "wavenumber-energy_spectra/$(Re)")
	kr, _ = energySpectrum(first(ζhs), grid)

	(mean, var) = statOver(ζhs) do ζh
		_, Ehr = energySpectrum(ζh, grid)
		Ehr
	end

	des["x"] = Array(kr)
	des["y"] = Array(mean)
	des["Δy"] = Array(sqrt.(var))

	return nothing
end

function energyFluxStat(Re)
	ζhs = getζhs(Re, false)
	des = create_group(desFile, "wavenumber-energy_fluxes/$(Re)")
	kr, _ = radialEnergyFlux(first(ζhs), grid)

	(mean, var) = statOver(ζhs) do ζh
		_, Ehr = radialEnergyFlux(ζh, grid)
		Ehr
	end

	des["x"] = Array(kr)
	des["y"] = Array(mean)
	des["Δy"] = Array(sqrt.(var))

	return nothing
end

function enstrophyFlusStat(Re)
	ζhs = getζhs(Re, false)
	des = create_group(desFile, "wavenumber-enstrophy_fluxes/$(Re)")
	kr, _ = radialEnstrophyFlux(first(ζhs), grid)

	(mean, var) = statOver(ζhs) do ζh
		_, Ehr = radialEnstrophyFlux(ζh, grid)
		Ehr
	end

	des["x"] = Array(kr)
	des["y"] = Array(mean)
	des["Δy"] = Array(sqrt.(var))

	return nothing
end

function velocityPDF(Re, isfiltered)
	ζhs = getζhs(Re, isfiltered)
	des = create_group(desFile, "velocity-pdf:filtered_$(isfiltered)/$(Re)")

	umax = maximum(ζhs) do ζh
		uh = im * ζh .* grid.invKrsq .* grid.l
		u = grid.rfftplan \ uh
		maximum(abs.(u))
	end

	bin = range(-umax, umax, 100)

	(mean, var) = statOver(ζhs) do ζh
		uh = im * ζh .* grid.invKrsq .* grid.l
		u = grid.rfftplan \ uh
		getPDF(u, bin)
	end

	des["x"] = Array((bin[1:(end-1)] .+ bin[2:end]) / 2)
	des["y"] = Array(mean)
	des["Δy"] = Array(sqrt.(var))

	return nothing
end

function rectLoopCircMoments_LoopSize(Re, order, width, height, ns)
	ζhs = getζhs(Re, true)
	des = create_group(desFile, "loop_sizes-rect_moments/$(Re)/$(order)/$(width)×$(height)")

	(mean, var) = statOver(ζhs) do ζh
		map(ns) do n
			Γ = rectCirculations(ζh, n*width, n*height, grid)
			sum(x->abs(x)^order, Γ) / length(Γ)
		end
	end

	des["x"] = Array(ns)
	des["y"] = Array(mean)
	des["Δy"] = Array(sqrt.(var))

	return nothing
end

function rectLoopCircMoments_AspectRatio_Area(Re, order, nloopsize)
	ζhs = getζhs(Re, true)
	des = create_group(desFile, "aspect_ratio-moments:equal area/$(Re)/$(nloopsize)/$(order)")
	ns = [n for n in 1:nloopsize if mod(nloopsize^2, n) == 0 && div(nloopsize^2, n) ≤ ngrid]

	(mean, var) = statOver(ζhs) do ζh
		map(ns) do n
			Γ = rectCirculations(ζh, n, div(nloopsize^2, n), grid)
			sum(x->abs(x)^order, Γ) / length(Γ)
		end
	end

	des["x"] = Array(ns)
	des["y"] = Array(mean)
	des["Δy"] = Array(sqrt.(var))

	return nothing
end

function rectLoopCircMoments_AspectRatio_Perimeter(Re, order, nloopsize)
	ζhs = getζhs(Re, true)
	des = create_group(desFile, "aspect_ratio-moments:equal perimeter/$(Re)/$(nloopsize)/$(order)")
	ns = [n for n in 1:nloopsize if 2nloopsize-n ≤ ngrid]

	(mean, var) = statOver(ζhs) do ζh
		map(ns) do n
			Γ = rectCirculations(ζh, n, 2nloopsize - n, grid)
			sum(x->abs(x)^order, Γ) / length(Γ)
		end
	end

	des["x"] = Array(ns)
	des["y"] = Array(mean)
	des["Δy"] = Array(sqrt.(var))

	return nothing
end

for Re in Reynolds
	@time energySpectrumStat(Re)
	@time energyFluxStat(Re)
	@time enstrophyFlusStat(Re)
	@time velocityPDF(Re, false)
	@time velocityPDF(Re, true)

	for order in 1:10
		@time rectLoopCircMoments_LoopSize(Re, order, 10, 10, 1:6)
		@time rectLoopCircMoments_LoopSize(Re, order, 5, 20, 1:6)
		@time rectLoopCircMoments_LoopSize(Re, order, 4, 16, 1:6)
		@time rectLoopCircMoments_AspectRatio_Area(Re, order, 20)
		@time rectLoopCircMoments_AspectRatio_Perimeter(Re, order, 20)
	end

end

close(desFile)
close.(values(srcFiles))








