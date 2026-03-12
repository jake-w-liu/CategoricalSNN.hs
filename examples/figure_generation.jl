## Publication Figure Generation for Categorical SNN Results
## Generates fig_results_*.pdf in paper/figs/

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
# Figure 1: LIF Neuron Dynamics — Cumulative spike counts
# ============================================================
println("Generating fig_results_lif_dynamics.pdf ...")
let
    csv = readdlm(joinpath(datadir, "experiment1_lif_neuron.csv"), ','; header=true)
    data = csv[1]
    t = Float64.(data[:, 1])
    lif_const = cumsum(Float64.(data[:, 3]))
    lif_bernoulli = cumsum(Float64.(data[:, 6]))
    baseline = cumsum(Float64.(data[:, 7]))

    fig = plot_scatter(t, lif_const;
        xlabel="Timestep", ylabel="Cumulative Spike Count",
        mode="lines", color=COLORS[1], dash=DASHES[1],
        legend="LIF (constant input)", linewidth=2)
    plot_scatter!(fig, t, lif_bernoulli;
        color=COLORS[2], dash=DASHES[2], mode="lines",
        legend="LIF (Bernoulli input)", linewidth=2)
    plot_scatter!(fig, t, baseline;
        color=COLORS[3], dash=DASHES[3], mode="lines",
        legend="Threshold baseline", linewidth=2)
    set_legend!(fig; position=:topleft)
    savefig(fig, joinpath(figdir, "fig_results_lif_dynamics.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    println("  Saved.")
end

# ============================================================
# Figure 2: Single-Layer Equivalence (Cat vs Flat)
# ============================================================
println("Generating fig_results_equivalence.pdf ...")
let
    csv = readdlm(joinpath(datadir, "experiment2_equivalence.csv"), ','; header=true)
    data = csv[1]
    t = Float64.(data[:, 1])
    cat0 = cumsum(Float64.(data[:, 2]))
    flat0 = cumsum(Float64.(data[:, 5]))
    cat1 = cumsum(Float64.(data[:, 3]))
    flat1 = cumsum(Float64.(data[:, 6]))
    cat2 = cumsum(Float64.(data[:, 4]))
    flat2 = cumsum(Float64.(data[:, 7]))

    fig = plot_scatter(t, cat0;
        xlabel="Timestep", ylabel="Cumulative Spike Count",
        mode="lines", color=COLORS[1], dash=DASHES[1],
        legend="Categorical (n0)", linewidth=2)
    plot_scatter!(fig, t, flat0;
        color=COLORS[2], dash=DASHES[2], mode="lines",
        legend="Flat (n0)", linewidth=2)
    plot_scatter!(fig, t, cat1;
        color=COLORS[3], dash=DASHES[3], mode="lines",
        legend="Categorical (n1)", linewidth=2)
    plot_scatter!(fig, t, flat1;
        color=COLORS[4], dash=DASHES[4], mode="lines",
        legend="Flat (n1)", linewidth=2)
    set_legend!(fig; position=:topleft)
    savefig(fig, joinpath(figdir, "fig_results_equivalence.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    println("  Saved.")
end

# ============================================================
# Figure 3: Two-Layer Network Composition
# ============================================================
println("Generating fig_results_network.pdf ...")
let
    csv = readdlm(joinpath(datadir, "experiment3_network.csv"), ','; header=true)
    data = csv[1]
    t = Float64.(data[:, 1])
    cat0 = cumsum(Float64.(data[:, 2]))
    flat0 = cumsum(Float64.(data[:, 4]))
    cat1 = cumsum(Float64.(data[:, 3]))
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
    savefig(fig, joinpath(figdir, "fig_results_network.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    println("  Saved.")
end

# ============================================================
# Figure 4: Tensor Product Independence
# ============================================================
println("Generating fig_results_tensor.pdf ...")
let
    csv = readdlm(joinpath(datadir, "experiment7_tensor.csv"), ','; header=true)
    data = csv[1]
    t = Float64.(data[:, 1])
    tensor_a = cumsum(Float64.(data[:, 4]))
    indep_a = cumsum(Float64.(data[:, 6]))
    tensor_b = cumsum(Float64.(data[:, 5]))
    indep_b = cumsum(Float64.(data[:, 7]))

    fig = plot_scatter(t, tensor_a;
        xlabel="Timestep", ylabel="Cumulative Spike Count",
        mode="lines", color=COLORS[1], dash=DASHES[1],
        legend="Tensor (path A)", linewidth=2)
    plot_scatter!(fig, t, indep_a;
        color=COLORS[2], dash=DASHES[2], mode="lines",
        legend="Independent (path A)", linewidth=2)
    plot_scatter!(fig, t, tensor_b;
        color=COLORS[3], dash=DASHES[3], mode="lines",
        legend="Tensor (path B)", linewidth=2)
    plot_scatter!(fig, t, indep_b;
        color=COLORS[4], dash=DASHES[4], mode="lines",
        legend="Independent (path B)", linewidth=2)
    set_legend!(fig; position=:topleft)
    savefig(fig, joinpath(figdir, "fig_results_tensor.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    println("  Saved.")
end

# ============================================================
# Figure 5: Robustness — Firing rates across weight configs
# ============================================================
println("Generating fig_results_robustness.pdf ...")
let
    csv = readdlm(joinpath(datadir, "experiment8_robustness.csv"), ','; header=true)
    data = csv[1]
    configs = String.(data[:, 1])
    rates0 = Float64.(data[:, 3])
    rates1 = Float64.(data[:, 4])
    rates2 = Float64.(data[:, 5])

    fig = plot_bar(configs, rates0;
        color=COLORS[1], legend="Neuron 1",
        ylabel="Firing Rate", xlabel="Weight Configuration")
    plot_bar!(fig, configs, rates1;
        color=COLORS[2], legend="Neuron 2")
    plot_bar!(fig, configs, rates2;
        color=COLORS[3], legend="Neuron 3")
    set_legend!(fig; position=:topright)
    savefig(fig, joinpath(figdir, "fig_results_robustness.pdf");
            width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
    println("  Saved.")
end

# ============================================================
# Figure 6: Synthesis scalability (generated only when data exists)
# ============================================================
let
    synth_csv = joinpath(datadir, "synthesis_results.csv")
    if isfile(synth_csv)
        csv = readdlm(synth_csv, ','; header=true)
        data = csv[1]
        if size(data, 1) > 0
            println("Generating fig_results_synthesis.pdf ...")
            configs = String.(data[:, 1])
            impls = String.(data[:, 2])
            luts = Float64.(data[:, 6])

            cfg_order = ["4to3", "4to3to2", "8to4", "16to8to4"]
            lut_cat = [luts[(configs .== cfg) .& (impls .== "categorical")][1] for cfg in cfg_order]
            lut_flat = [luts[(configs .== cfg) .& (impls .== "flat")][1] for cfg in cfg_order]

            fig = plot_bar(cfg_order, lut_cat;
                color=COLORS[1], legend="Categorical",
                ylabel="LUT Count", xlabel="Network Configuration")
            plot_bar!(fig, cfg_order, lut_flat;
                color=COLORS[2], legend="Flat")
            set_legend!(fig; position=:topleft)
            savefig(fig, joinpath(figdir, "fig_results_synthesis.pdf");
                    width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_H)
            println("  Saved.")
        else
            println("Skipping fig_results_synthesis.pdf (no synthesis rows yet).")
        end
    else
        println("Skipping fig_results_synthesis.pdf (synthesis_results.csv not found).")
    end
end

println("\nAll publication figures generated.")
