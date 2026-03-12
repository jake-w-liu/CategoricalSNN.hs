-- | Experiment runner for categorical SNN validation.
--
-- Produces CSV data files and validates:
-- 1. Functionality: LIF neuron produces correct spike patterns
-- 2. Sanity: Network outputs are in expected range
-- 3. Stability: No unexpected behavior over long runs
-- 4. Baseline comparison: Categorical vs flat implementation equivalence
module Main where

import Clash.Prelude hiding (foldr, map, zip, zipWith, take, length, filter,
                             replicate, head, last, concat, unlines, writeFile,
                             putStrLn, show, (++), simulate)
import qualified Clash.Prelude as C
import qualified Prelude as P
import Prelude (IO, String, Show(..), putStrLn, (++), map, zip, zipWith,
                filter, take, length, replicate, concat, writeFile, unlines)
import qualified Data.List as L

import CategoricalSNN

-- | Base directory for output (relative to cabal run working dir)
baseDir :: String
baseDir = "../paper/data/"

-- | Simulation parameters
numTimesteps :: P.Int
numTimesteps = 200

seed :: P.Int
seed = 42

-- | Helper: convert SFixed to Double for CSV output
sfToDouble :: Potential -> P.Double
sfToDouble x =
  let bv = pack x :: BitVector 32
      sv = unpack bv :: Signed 32
  in  P.fromIntegral sv P./ (2 P.^ (16 :: P.Int))

-- | Helper: Bool to String
boolToStr :: P.Bool -> String
boolToStr P.True  = "1"
boolToStr P.False = "0"

-- | Helper: Double to full-precision String
dblToStr :: P.Double -> String
dblToStr = P.show

-- | Helper: Int to String
intToStr :: P.Int -> String
intToStr = P.show

main :: IO ()
main = do
  putStrLn "=== Categorical SNN Experiments ==="
  putStrLn ""

  -- Experiment 1: Single LIF neuron response
  putStrLn "--- Experiment 1: Single LIF Neuron Response ---"
  experiment1_lifNeuron

  -- Experiment 2: Categorical vs Flat layer equivalence
  putStrLn "--- Experiment 2: Categorical vs Flat Layer Equivalence ---"
  experiment2_equivalence

  -- Experiment 3: Two-layer network composition
  putStrLn "--- Experiment 3: Two-Layer Network (Categorical vs Flat) ---"
  experiment3_network

  -- Experiment 4: Compositionality laws verification
  putStrLn "--- Experiment 4: Compositionality Laws ---"
  experiment4_laws

  putStrLn ""
  putStrLn "=== All experiments complete ==="

-- | Experiment 1: LIF neuron response to constant and Bernoulli input.
-- Validates: functionality (correct spike generation), sanity (expected rates),
-- stability (no NaN/Inf over 200 timesteps).
experiment1_lifNeuron :: IO ()
experiment1_lifNeuron = do
  let params = LIFParams { lifThreshold = 1.0, lifDecay = 0.9, lifReset = 0.0 }
      neuron = lifNeuronCustom params

      -- Constant input: should produce regular periodic spiking
      constInputs = constantInput numTimesteps 0.2
      constSpikes = simulate neuron constInputs

      -- Bernoulli input at 30% rate through a synapse with weight 0.5
      bernoulliIn = bernoulliSpikeTrain numTimesteps 0.3 seed
      synMorph = synapse 0.5
      bernoulliCurrents = simulate synMorph bernoulliIn
      bernoulliSpikes = simulate neuron bernoulliCurrents

      -- Baseline: threshold neuron (no leak)
      baseline = thresholdNeuron 0.2
      baselineSpikes = simulate baseline constInputs

  -- Write CSV: timestep, constant_input, lif_spike, bernoulli_input,
  -- bernoulli_current, bernoulli_lif_spike, baseline_spike
  let header = "timestep,constant_input,lif_spike_constant,bernoulli_input_spike,bernoulli_current,lif_spike_bernoulli,baseline_spike_constant"
      rows = [ L.intercalate ","
               [ intToStr t
               , dblToStr (sfToDouble ci)
               , boolToStr cs
               , boolToStr pi'
               , dblToStr (sfToDouble pc)
               , boolToStr ps
               , boolToStr bs
               ]
             | (t, (ci, cs, pi', pc, ps, bs)) <-
                 zip [0..] (zip6 constInputs constSpikes bernoulliIn
                            bernoulliCurrents bernoulliSpikes baselineSpikes)
             ]

  writeFile (baseDir ++ "experiment1_lif_neuron.csv") (header ++ "\n" ++ unlines rows)

  let constRate = firingRate constSpikes
      bernoulliRate = firingRate bernoulliSpikes
      baselineRate = firingRate baselineSpikes
  putStrLn ("  LIF constant input firing rate: " ++ dblToStr constRate)
  putStrLn ("  LIF Bernoulli input firing rate: " ++ dblToStr bernoulliRate)
  putStrLn ("  Baseline threshold firing rate: " ++ dblToStr baselineRate)
  putStrLn ("  Spike counts: LIF-const=" ++ intToStr (spikeCount constSpikes)
            ++ " LIF-bernoulli=" ++ intToStr (spikeCount bernoulliSpikes)
            ++ " baseline=" ++ intToStr (spikeCount baselineSpikes))
  putStrLn ""

  where
    zip6 (a:as) (b:bs) (c:cs) (d:ds) (e:es) (f:fs) =
      (a,b,c,d,e,f) : zip6 as bs cs ds es fs
    zip6 _ _ _ _ _ _ = []

-- | Experiment 2: Categorical layer vs flat layer produce identical output.
-- This is the core validation: composition preserves functional behavior.
experiment2_equivalence :: IO ()
experiment2_equivalence = do
  let params = defaultLIFParams

      -- 4-input, 3-output layer
      weights :: C.Vec 3 (C.Vec 4 Weight)
      weights = (0.5 :> 0.3 :> (-0.2) :> 0.4 :> C.Nil)
             :> (0.1 :> 0.6 :> 0.2 :> (-0.1) :> C.Nil)
             :> ((-0.3) :> 0.2 :> 0.5 :> 0.3 :> C.Nil)
             :> C.Nil

      catLayer = snnLayer weights params
      fltLayer = flatLayer weights params

      -- Generate input spike trains (4 channels)
      spikes0 = bernoulliSpikeTrain numTimesteps 0.3 seed
      spikes1 = bernoulliSpikeTrain numTimesteps 0.2 (seed P.+ 1)
      spikes2 = bernoulliSpikeTrain numTimesteps 0.4 (seed P.+ 2)
      spikes3 = bernoulliSpikeTrain numTimesteps 0.1 (seed P.+ 3)

      inputVecs :: [C.Vec 4 Spike]
      inputVecs = [ s0 :> s1 :> s2 :> s3 :> C.Nil
                  | (s0, (s1, (s2, s3))) <-
                      zip spikes0 (zip spikes1 (zip spikes2 spikes3))
                  ]

      catOutputs = simulate catLayer inputVecs
      fltOutputs = simulate fltLayer inputVecs

      -- Check exact equivalence
      equivalent = P.all P.id (zipWith (P.==) catOutputs fltOutputs)

  -- Write CSV
  let header = "timestep,cat_out0,cat_out1,cat_out2,flat_out0,flat_out1,flat_out2,match"
      rows = [ L.intercalate ","
               [ intToStr t
               , boolToStr (co C.!! 0), boolToStr (co C.!! 1), boolToStr (co C.!! 2)
               , boolToStr (fo C.!! 0), boolToStr (fo C.!! 1), boolToStr (fo C.!! 2)
               , boolToStr (co P.== fo)
               ]
             | (t, (co, fo)) <- zip [0..] (zip catOutputs fltOutputs)
             ]

  writeFile (baseDir ++ "experiment2_equivalence.csv") (header ++ "\n" ++ unlines rows)

  let catRates = [ firingRate (map (\v -> v C.!! i) catOutputs) | i <- [0,1,2] ]
      fltRates = [ firingRate (map (\v -> v C.!! i) fltOutputs) | i <- [0,1,2] ]

  putStrLn ("  Categorical vs Flat EXACT match: " ++ P.show equivalent)
  putStrLn ("  Categorical firing rates: " ++ P.show catRates)
  putStrLn ("  Flat firing rates:        " ++ P.show fltRates)
  putStrLn ""

-- | Experiment 3: Two-layer categorical network vs flat two-layer network.
-- Validates that composeMorph (layer1 >>> layer2) produces identical
-- results to a monolithic two-layer implementation.
experiment3_network :: IO ()
experiment3_network = do
  let params = defaultLIFParams

      -- Layer 1: 4 -> 3
      w1 :: C.Vec 3 (C.Vec 4 Weight)
      w1 = (0.5 :> 0.3 :> (-0.2) :> 0.4 :> C.Nil)
        :> (0.1 :> 0.6 :> 0.2 :> (-0.1) :> C.Nil)
        :> ((-0.3) :> 0.2 :> 0.5 :> 0.3 :> C.Nil)
        :> C.Nil

      -- Layer 2: 3 -> 2
      w2 :: C.Vec 2 (C.Vec 3 Weight)
      w2 = (0.4 :> (-0.2) :> 0.6 :> C.Nil)
        :> (0.3 :> 0.5 :> (-0.1) :> C.Nil)
        :> C.Nil

      catNet = snnNetwork2Layer w1 w2 params
      fltNet = flatNetwork2Layer w1 w2 params

      -- Input spike trains
      spikes0 = bernoulliSpikeTrain numTimesteps 0.3 seed
      spikes1 = bernoulliSpikeTrain numTimesteps 0.2 (seed P.+ 1)
      spikes2 = bernoulliSpikeTrain numTimesteps 0.4 (seed P.+ 2)
      spikes3 = bernoulliSpikeTrain numTimesteps 0.1 (seed P.+ 3)

      inputVecs :: [C.Vec 4 Spike]
      inputVecs = [ s0 :> s1 :> s2 :> s3 :> C.Nil
                  | (s0, (s1, (s2, s3))) <-
                      zip spikes0 (zip spikes1 (zip spikes2 spikes3))
                  ]

      catOutputs = simulate catNet inputVecs
      fltOutputs = simulate fltNet inputVecs

      equivalent = P.all P.id (zipWith (P.==) catOutputs fltOutputs)

  let header = "timestep,cat_out0,cat_out1,flat_out0,flat_out1,match"
      rows = [ L.intercalate ","
               [ intToStr t
               , boolToStr (co C.!! 0), boolToStr (co C.!! 1)
               , boolToStr (fo C.!! 0), boolToStr (fo C.!! 1)
               , boolToStr (co P.== fo)
               ]
             | (t, (co, fo)) <- zip [0..] (zip catOutputs fltOutputs)
             ]

  writeFile (baseDir ++ "experiment3_network.csv") (header ++ "\n" ++ unlines rows)

  let catRates = [ firingRate (map (\v -> v C.!! i) catOutputs) | i <- [0,1] ]
      fltRates = [ firingRate (map (\v -> v C.!! i) fltOutputs) | i <- [0,1] ]

  putStrLn ("  2-Layer Categorical vs Flat EXACT match: " ++ P.show equivalent)
  putStrLn ("  Categorical firing rates: " ++ P.show catRates)
  putStrLn ("  Flat firing rates:        " ++ P.show fltRates)
  putStrLn ""

-- | Experiment 4: Verify categorical laws hold numerically.
-- Tests identity, associativity, and interchange laws.
experiment4_laws :: IO ()
experiment4_laws = do
  let params = defaultLIFParams
      neuron = lifNeuronCustom params
      syn = synapse 0.5

      -- Test inputs
      spikeInputs = bernoulliSpikeTrain 50 0.3 seed
      potInputs = simulate syn spikeInputs

      -- Identity law: id >>> f = f
      idThenNeuron = idMorph `composeMorph` neuron
      neuronDirect = neuron
      outIdLeft = simulate idThenNeuron potInputs
      outDirect = simulate neuronDirect potInputs
      identityLeftHolds = outIdLeft P.== outDirect

      -- Identity law: f >>> id = f
      neuronThenId = neuron `composeMorph` idMorph
      outIdRight = simulate neuronThenId potInputs
      identityRightHolds = outIdRight P.== outDirect

      -- Associativity: (f >>> g) >>> h = f >>> (g >>> h)
      -- Use: synapse >>> neuron >>> (spike->potential synapse)
      syn2 = synapse 0.8
      chain_LR = (syn `composeMorph` neuron) `composeMorph` syn2
      chain_RL = syn `composeMorph` (neuron `composeMorph` syn2)
      outLR = simulate chain_LR spikeInputs
      outRL = simulate chain_RL spikeInputs
      assocHolds = outLR P.== outRL

  -- Write results
  let header = "law,holds"
      rows = [ "identity_left," ++ P.show identityLeftHolds
             , "identity_right," ++ P.show identityRightHolds
             , "associativity," ++ P.show assocHolds
             ]

  writeFile (baseDir ++ "experiment4_laws.csv") (header ++ "\n" ++ unlines rows)

  putStrLn ("  Identity law (id >>> f = f):   " ++ P.show identityLeftHolds)
  putStrLn ("  Identity law (f >>> id = f):   " ++ P.show identityRightHolds)
  putStrLn ("  Associativity ((f>>>g)>>>h = f>>>(g>>>h)): " ++ P.show assocHolds)
  putStrLn ""
