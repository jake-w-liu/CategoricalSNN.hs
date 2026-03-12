# Plot results from CategoricalSNN experiments.
# Run AFTER the Haskell experiments executable produces CSV files.
#
# Uses PlotlySupply.jl (see plotting-guide.md for API).

using PlotlySupply, DelimitedFiles
import PlotlyKaleido
import PlotlySupply: savefig

PlotlyKaleido.start(mathjax=false)

const COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
const DASHES = ["solid", "dash", "dashdot", "dot"]
const IEEE_SINGLE_COL_W = 504
const IEEE_SINGLE_COL_H = 360

datadir = joinpath(@__DIR__, "..", "..", "paper", "data")
figdir  = joinpath(@__DIR__, "..", "..", "paper", "figs")
mkpath(figdir)

# ============================================================
# Plot 1: LIF Neuron Response — spike raster + membrane proxy
# ============================================================
println("Plotting heuristic_lif_response...")
let
    csv = readdlm(joinpath(datadir, "experiment1_lif_neuron.csv"), ','; header=true)
    data = csv[1]
    # Columns: timestep, constant_input, lif_spike_constant,
    #          bernoulli_input_spike, bernoulli_current, lif_spike_bernoulli,
    #          baseline_spike_constant
    t   = Float64.(data[:, 1])
    lif_const   = Float64.(data[:, 3])
    lif_bernoulli = Float64.(data[:, 6])
    baseline    = Float64.(data[:, 7])

    # Cumulative spike count (shows integration behavior over time)
    cum_lif_const   = cumsum(lif_const)
    cum_lif_bernoulli = cumsum(lif_bernoulli)
    cum_baseline    = cumsum(baseline)

    fig = plot_scatter(t, cum_lif_const;
        xlabel="Timestep", ylabel="Cumulative Spike Count",
        mode="lines", color=COLORS[1], dash=DASHES[1],
        legend="LIF (constant input)", linewidth=2)
    plot_scatter!(fig, t, cum_lif_bernoulli;
        color=COLORS[2], dash=DASHES[2], mode="lines",
        legend="LIF (Bernoulli input)", linewidth=2)
    plot_scatter!(fig, t, cum_baseline;
        color=COLORS[3], dash=DASHES[3], mode="lines",
        legend="Threshold baseline (constant)", linewidth=2)
    set_legend!(fig; position=:topright)
    savefig(fig, joinpath(figdir, "heuristic_lif_response.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    println("  Saved heuristic_lif_response.pdf")
end

# ============================================================
# Plot 2: Categorical vs Flat Layer Equivalence
# ============================================================
println("Plotting heuristic_equivalence...")
let
    csv = readdlm(joinpath(datadir, "experiment2_equivalence.csv"), ','; header=true)
    data = csv[1]
    t = Float64.(data[:, 1])
    # Show cumulative spikes for categorical output 0 vs flat output 0
    cat0 = cumsum(Float64.(data[:, 2]))
    flat0 = cumsum(Float64.(data[:, 5]))
    cat1 = cumsum(Float64.(data[:, 3]))
    flat1 = cumsum(Float64.(data[:, 6]))

    fig = plot_scatter(t, cat0;
        xlabel="Timestep", ylabel="Cumulative Spike Count",
        mode="lines", color=COLORS[1], dash=DASHES[1],
        legend="Categorical (neuron 0)", linewidth=2)
    plot_scatter!(fig, t, flat0;
        color=COLORS[2], dash=DASHES[2], mode="lines",
        legend="Flat (neuron 0)", linewidth=2)
    plot_scatter!(fig, t, cat1;
        color=COLORS[3], dash=DASHES[3], mode="lines",
        legend="Categorical (neuron 1)", linewidth=2)
    plot_scatter!(fig, t, flat1;
        color=COLORS[4], dash=DASHES[4], mode="lines",
        legend="Flat (neuron 1)", linewidth=2)
    set_legend!(fig; position=:topleft)
    savefig(fig, joinpath(figdir, "heuristic_equivalence.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    println("  Saved heuristic_equivalence.pdf")
end

# ============================================================
# Plot 3: Two-Layer Network — Categorical vs Flat Composition
# ============================================================
println("Plotting heuristic_network_composition...")
let
    csv = readdlm(joinpath(datadir, "experiment3_network.csv"), ','; header=true)
    data = csv[1]
    t = Float64.(data[:, 1])
    cat0 = cumsum(Float64.(data[:, 2]))
    cat1 = cumsum(Float64.(data[:, 3]))
    flat0 = cumsum(Float64.(data[:, 4]))
    flat1 = cumsum(Float64.(data[:, 5]))

    fig = plot_scatter(t, cat0;
        xlabel="Timestep", ylabel="Cumulative Spike Count",
        mode="lines", color=COLORS[1], dash=DASHES[1],
        legend="Categorical (out 0)", linewidth=2)
    plot_scatter!(fig, t, flat0;
        color=COLORS[2], dash=DASHES[2], mode="lines",
        legend="Flat (out 0)", linewidth=2)
    plot_scatter!(fig, t, cat1;
        color=COLORS[3], dash=DASHES[3], mode="lines",
        legend="Categorical (out 1)", linewidth=2)
    plot_scatter!(fig, t, flat1;
        color=COLORS[4], dash=DASHES[4], mode="lines",
        legend="Flat (out 1)", linewidth=2)
    set_legend!(fig; position=:topleft)
    savefig(fig, joinpath(figdir, "heuristic_network_composition.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    println("  Saved heuristic_network_composition.pdf")
end

println("\nAll plots generated.")
