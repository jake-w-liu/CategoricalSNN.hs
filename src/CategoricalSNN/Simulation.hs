-- | Pure-Haskell simulation engine for categorical SNN circuits.
--
-- This module provides functions to run SNN morphisms over input
-- spike trains and collect output traces for analysis and plotting.
-- Results are exported as CSV for Julia/PlotlySupply visualization.
module CategoricalSNN.Simulation
  ( -- * Simulation
    simulate
  , simulateWithTrace
    -- * Input generation
  , bernoulliSpikeTrain
  , periodicSpikeTrain
  , constantInput
    -- * Output analysis
  , spikeCount
  , firingRate
  , spikeTimestamps
    -- * CSV export
  , writeCSV
  , writeCSVWithHeader
  ) where

import Clash.Prelude hiding (foldr, (^), simulate)
import qualified Prelude as P
import Prelude ((^))
import qualified Data.List as L

-- Import Category for runMorph
import CategoricalSNN.Category (SNNMorphism(..), runMorph)
import CategoricalSNN.Types

-- | Simulate an SNN morphism over a list of inputs.
-- Returns the list of outputs at each timestep.
simulate :: SNNMorphism a b -> [a] -> [b]
simulate = runMorph

-- | Simulate with full state trace: returns (outputs, final_morphism).
-- Useful for inspecting intermediate states.
simulateWithTrace :: SNNMorphism a b -> [a] -> ([b], SNNMorphism a b)
simulateWithTrace m [] = ([], m)
simulateWithTrace (SNNMorphism f s0) (x:xs) =
  let (s1, y) = f s0 x
      (ys, mFinal) = simulateWithTrace (SNNMorphism f s1) xs
  in  (y : ys, mFinal)

-- | Generate a Bernoulli spike train using a linear congruential generator.
-- rate: average firing probability per timestep (0.0 to 1.0).
-- Uses deterministic PRNG with seed for reproducibility.
bernoulliSpikeTrain :: Int      -- ^ Number of timesteps
                  -> Double   -- ^ Firing rate (probability per step)
                  -> Int      -- ^ Random seed
                  -> [Spike]
bernoulliSpikeTrain n rate seed =
  let -- Simple LCG: x_{n+1} = (a*x_n + c) mod m
      lcg x = (1103515245 * x + 12345) `P.mod` (2^(31 :: Int))
      rands = P.take n (P.iterate lcg seed)
      threshold = P.floor (rate P.* P.fromIntegral (2^(31 :: Int) :: Int)) :: Int
  in  P.map (P.< threshold) rands

-- | Generate a periodic spike train with given period.
-- Spikes at timesteps 0, period, 2*period, ...
periodicSpikeTrain :: Int    -- ^ Number of timesteps
                   -> Int    -- ^ Period (timesteps between spikes)
                   -> [Spike]
periodicSpikeTrain n period =
  [t `P.mod` period P.== 0 | t <- [0..n P.-1]]

-- | Constant input: a fixed potential value at every timestep.
constantInput :: Int -> Potential -> [Potential]
constantInput n v = P.replicate n v

-- | Count total spikes in a trace.
spikeCount :: [Spike] -> Int
spikeCount = P.length P.. P.filter P.id

-- | Compute firing rate: spikes / total timesteps.
firingRate :: [Spike] -> Double
firingRate spikes =
  let total = P.length spikes
      fired = spikeCount spikes
  in  if total P.== 0 then 0.0
      else P.fromIntegral fired P./ P.fromIntegral total

-- | Extract timestep indices where spikes occurred.
spikeTimestamps :: [Spike] -> [Int]
spikeTimestamps spikes = [t | (t, True) <- P.zip [0..] spikes]

-- | Write data to CSV file with custom header.
writeCSVWithHeader :: P.FilePath -> P.String -> [[P.String]] -> P.IO ()
writeCSVWithHeader path header rows = do
  let content = header P.++ "\n" P.++ P.unlines (P.map (L.intercalate ",") rows)
  P.writeFile path content

-- | Write data to CSV with auto-generated column names.
writeCSV :: P.FilePath -> [P.String] -> [[P.String]] -> P.IO ()
writeCSV path columns rows =
  writeCSVWithHeader path (L.intercalate "," columns) rows
