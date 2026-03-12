-- | Comprehensive test suite for CategoricalSNN.
--
-- Tests cover:
-- 1. Core algorithm correctness (LIF neuron against analytical solutions)
-- 2. Categorical law verification (identity, associativity, interchange)
-- 3. Categorical-vs-flat equivalence (single layer and multi-layer)
-- 4. Edge cases and input validation
-- 5. Consistency of outputs (dimensions, types, no NaN/Inf behavior)
module Main where

import Clash.Prelude hiding (map, zip, zipWith, filter, length, replicate,
                             take, head, last, concat, (++), simulate)
import qualified Clash.Prelude as C
import qualified Prelude as P
import Prelude (IO, Bool(..), (==), (&&), (||), not, map, zip, zipWith,
                filter, length, all, any, ($), (.), putStrLn, (++), show,
                String, Int, Double)

import Test.Tasty
import Test.Tasty.HUnit

import CategoricalSNN

main :: IO ()
main = defaultMain tests

tests :: TestTree
tests = testGroup "CategoricalSNN"
  [ testGroup "LIF Neuron Correctness" lifTests
  , testGroup "Synapse Correctness" synapseTests
  , testGroup "Categorical Laws" lawTests
  , testGroup "Layer Equivalence" equivalenceTests
  , testGroup "Network Composition" networkTests
  , testGroup "Edge Cases" edgeTests
  , testGroup "Output Consistency" consistencyTests
  ]

-- ============================================================
-- 1. LIF Neuron Correctness
-- ============================================================

lifTests :: [TestTree]
lifTests =
  [ testCase "LIF neuron fires when threshold exceeded" $ do
      -- With decay=0.9, threshold=1.0, constant input=0.2:
      -- V accumulates until crossing 1.0, then fires
      let params = LIFParams 1.0 0.9 0.0
          neuron = lifNeuronCustom params
          inputs = P.replicate 20 (0.2 :: Potential)
          spikes = simulate neuron inputs
          firstSpike = P.head (spikeTimestamps spikes)
      -- Should fire around timestep 6-8
      assertBool "First spike within expected range" (firstSpike P.>= 5 P.&& firstSpike P.<= 8)

  , testCase "LIF neuron does not fire below threshold" $ do
      -- Very small input with high threshold: should never fire
      let params = LIFParams 10.0 0.5 0.0
          neuron = lifNeuronCustom params
          inputs = P.replicate 100 (0.01 :: Potential)
          spikes = simulate neuron inputs
      -- With decay=0.5 and input=0.01, V converges to 0.02 << 10
      assertEqual "No spikes" 0 (spikeCount spikes)

  , testCase "LIF neuron resets after firing" $ do
      let params = LIFParams 1.0 0.9 0.0
          neuron = lifNeuronCustom params
          -- Large input to guarantee immediate spike, then zeros
          inputs = [2.0 :: Potential, 0.0, 0.0, 0.0, 0.0]
          spikes = simulate neuron inputs
      assertEqual "Fires at t=0" True (P.head spikes)
      assertEqual "Silent after reset" 0 (spikeCount (P.tail spikes))

  , testCase "LIF decay is correct (analytical)" $ do
      -- With no input, V(t) = V_prev * decay
      let params = LIFParams 1.0 0.8 0.5
          (v1, _) = lifStep params 0.5 0.0  -- start at 0.5, no input
          (v2, _) = lifStep params v1 0.0
      -- V1 = 0.5 * 0.8 = 0.4; tolerance for fixed-point
      assertBool "Decay step 1" (v1 P.> 0.39 P.&& v1 P.< 0.41)
      -- V2 = 0.4 * 0.8 = 0.32
      assertBool "Decay step 2" (v2 P.> 0.31 P.&& v2 P.< 0.33)

  , testCase "Threshold neuron baseline fires on every above-threshold input" $ do
      let baseline = thresholdNeuron 0.5
          inputs = [0.3, 0.7, 0.2, 0.9, 0.5] :: [Potential]
          spikes = simulate baseline inputs
      assertEqual "Correct spike pattern" [False, True, False, True, True] spikes
  ]

-- ============================================================
-- 2. Synapse Correctness
-- ============================================================

synapseTests :: [TestTree]
synapseTests =
  [ testCase "Synapse multiplies spike by weight" $ do
      let syn = synapse 0.5
          inputs = [True, False, True, False] :: [Spike]
          outputs = simulate syn inputs
      assertBool "Spike produces weight" (P.head outputs P.> 0.49)
      assertEqual "No spike produces zero" 0.0 (outputs P.!! 1)

  , testCase "Wire synapse produces 1.0 for spike" $ do
      let syn = wireSynapse
          outputs = simulate syn [True, False]
      assertBool "Wire outputs ~1.0" (P.head outputs P.> 0.99)
      assertEqual "Wire outputs 0.0" 0.0 (outputs P.!! 1)

  , testCase "Negative weight synapse" $ do
      let syn = synapse (-0.5)
          outputs = simulate syn [True]
      assertBool "Negative output" (P.head outputs P.< 0.0)

  , testCase "Delayed synapse respects delay 1" $
      assertDelayTap 1 [False, True, False, False, False, False, False, False, False]

  , testCase "Delayed synapse respects delay 3" $
      assertDelayTap 3 [False, False, False, True, False, False, False, False, False]

  , testCase "Delayed synapse respects delay 7" $
      assertDelayTap 7 [False, False, False, False, False, False, False, True, False]
  ]

assertDelayTap :: Index 8 -> [Bool] -> Assertion
assertDelayTap delay expected = do
  let syn = synapseWithDelay 0.5 delay
      inputs = True : P.replicate 8 False
      outputs = simulate syn inputs
      fired = map (> 0.49) outputs
  assertEqual ("Delay tap " ++ show delay) expected fired

-- ============================================================
-- 3. Categorical Laws
-- ============================================================

lawTests :: [TestTree]
lawTests =
  [ testCase "Left identity: id >>> f = f" $ do
      let f = lifNeuronCustom defaultLIFParams
          idF = idMorph `composeMorph` f
          inputs = constantInput 50 0.2
      assertEqual "Left identity holds" (simulate f inputs) (simulate idF inputs)

  , testCase "Right identity: f >>> id = f" $ do
      let f = lifNeuronCustom defaultLIFParams
          fId = f `composeMorph` idMorph
          inputs = constantInput 50 0.2
      assertEqual "Right identity holds" (simulate f inputs) (simulate fId inputs)

  , testCase "Associativity: (f >>> g) >>> h = f >>> (g >>> h)" $ do
      let f = synapse 0.5
          g = lifNeuronCustom defaultLIFParams
          h = synapse 0.8
          inputs = bernoulliSpikeTrain 50 0.3 42
          lAssoc = (f `composeMorph` g) `composeMorph` h
          rAssoc = f `composeMorph` (g `composeMorph` h)
      assertEqual "Associativity holds" (simulate lAssoc inputs) (simulate rAssoc inputs)

  , testCase "Tensor product independence" $ do
      let f = synapse 0.5
          g = synapse 0.3
          fg = tensorMorph f g
          inputs = [(True, False), (False, True), (True, True)]
          outputs = simulate fg inputs
      assertBool "Tensor first component" (fst (P.head outputs) P.> 0.49)
      assertEqual "Tensor second component zero" 0.0 (snd (P.head outputs))
      assertEqual "Tensor first zero" 0.0 (fst (outputs P.!! 1))
      assertBool "Tensor second component" (snd (outputs P.!! 1) P.> 0.29)

  , testCase "Braiding: braid >>> braid = id" $ do
      let braidBraid = braidMorph `composeMorph` braidMorph
          inputs = [(1 :: Potential, 2 :: Potential), (3, 4), (5, 6)]
      assertEqual "Double braid is identity" inputs (simulate braidBraid inputs)
  ]

-- ============================================================
-- 4. Layer Equivalence (Categorical vs Flat)
-- ============================================================

equivalenceTests :: [TestTree]
equivalenceTests =
  [ testCase "Single layer: categorical = flat (exact)" $ do
      let weights :: C.Vec 3 (C.Vec 4 Weight)
          weights = (0.5 :> 0.3 :> (-0.2) :> 0.4 :> C.Nil)
                 :> (0.1 :> 0.6 :> 0.2 :> (-0.1) :> C.Nil)
                 :> ((-0.3) :> 0.2 :> 0.5 :> 0.3 :> C.Nil)
                 :> C.Nil
          params = defaultLIFParams
          catL = snnLayer weights params
          fltL = flatLayer weights params
          inputs = [ s0 :> s1 :> s2 :> s3 :> C.Nil
                   | (s0, (s1, (s2, s3))) <-
                       zip (bernoulliSpikeTrain 100 0.3 42)
                           (zip (bernoulliSpikeTrain 100 0.2 43)
                                (zip (bernoulliSpikeTrain 100 0.4 44)
                                     (bernoulliSpikeTrain 100 0.1 45)))
                   ]
      assertEqual "Single layer exact match" (simulate catL inputs) (simulate fltL inputs)

  , testCase "Layer with all-zero weights produces no spikes" $ do
      let weights :: C.Vec 2 (C.Vec 3 Weight)
          weights = (0 :> 0 :> 0 :> C.Nil)
                 :> (0 :> 0 :> 0 :> C.Nil)
                 :> C.Nil
          layer = snnLayer weights defaultLIFParams
          inputs = P.replicate 50 (True :> True :> True :> C.Nil)
          outputs = simulate layer inputs
          totalSpikes = P.sum (map (\v -> (if v C.!! 0 then 1 else 0)
                                       P.+ (if v C.!! 1 then 1 else 0)) outputs) :: Int
      assertEqual "No spikes with zero weights" 0 totalSpikes
  ]

-- ============================================================
-- 5. Network Composition
-- ============================================================

networkTests :: [TestTree]
networkTests =
  [ testCase "Two-layer categorical = flat (exact)" $ do
      let w1 :: C.Vec 3 (C.Vec 4 Weight)
          w1 = (0.5 :> 0.3 :> (-0.2) :> 0.4 :> C.Nil)
            :> (0.1 :> 0.6 :> 0.2 :> (-0.1) :> C.Nil)
            :> ((-0.3) :> 0.2 :> 0.5 :> 0.3 :> C.Nil)
            :> C.Nil
          w2 :: C.Vec 2 (C.Vec 3 Weight)
          w2 = (0.4 :> (-0.2) :> 0.6 :> C.Nil)
            :> (0.3 :> 0.5 :> (-0.1) :> C.Nil)
            :> C.Nil
          params = defaultLIFParams
          catNet = snnNetwork2Layer w1 w2 params
          fltNet = flatNetwork2Layer w1 w2 params
          inputs = [ s0 :> s1 :> s2 :> s3 :> C.Nil
                   | (s0, (s1, (s2, s3))) <-
                       zip (bernoulliSpikeTrain 100 0.3 42)
                           (zip (bernoulliSpikeTrain 100 0.2 43)
                                (zip (bernoulliSpikeTrain 100 0.4 44)
                                     (bernoulliSpikeTrain 100 0.1 45)))
                   ]
      assertEqual "Two-layer exact match" (simulate catNet inputs) (simulate fltNet inputs)

  , testCase "Three-layer irregular categorical = flat (exact)" $ do
      let w12to7 :: C.Vec 7 (C.Vec 12 Weight)
          w12to7 =
               ( 0.25 :> (-0.10) :> 0.15 :> 0.00 :> 0.20 :> (-0.05) :> 0.10 :> 0.30 :> (-0.12) :> 0.18 :> 0.00 :> 0.22 :> C.Nil)
            :> ((-0.18) :> 0.24 :> 0.00 :> 0.12 :> 0.08 :> 0.16 :> (-0.14) :> 0.28 :> 0.11 :> 0.00 :> 0.20 :> (-0.06) :> C.Nil)
            :> ( 0.05 :> 0.19 :> (-0.22) :> 0.31 :> 0.00 :> 0.14 :> 0.26 :> (-0.09) :> 0.17 :> 0.07 :> (-0.04) :> 0.12 :> C.Nil)
            :> ( 0.29 :> 0.00 :> 0.13 :> (-0.17) :> 0.21 :> 0.09 :> 0.00 :> 0.24 :> (-0.08) :> 0.16 :> 0.06 :> (-0.11) :> C.Nil)
            :> ((-0.07) :> 0.27 :> 0.18 :> 0.00 :> (-0.15) :> 0.23 :> 0.12 :> 0.05 :> 0.19 :> (-0.10) :> 0.14 :> 0.00 :> C.Nil)
            :> ( 0.11 :> 0.04 :> 0.25 :> (-0.13) :> 0.28 :> 0.00 :> 0.09 :> (-0.16) :> 0.22 :> 0.15 :> 0.03 :> 0.18 :> C.Nil)
            :> ( 0.00 :> 0.21 :> (-0.05) :> 0.18 :> 0.24 :> 0.10 :> (-0.09) :> 0.13 :> 0.00 :> 0.20 :> 0.16 :> (-0.12) :> C.Nil)
            :> C.Nil
          w7to5 :: C.Vec 5 (C.Vec 7 Weight)
          w7to5 =
               ( 0.30 :> (-0.18) :> 0.22 :> 0.00 :> 0.14 :> 0.19 :> (-0.11) :> C.Nil)
            :> ((-0.09) :> 0.27 :> 0.16 :> 0.21 :> 0.00 :> (-0.13) :> 0.24 :> C.Nil)
            :> ( 0.18 :> 0.00 :> (-0.15) :> 0.26 :> 0.12 :> 0.20 :> 0.07 :> C.Nil)
            :> ( 0.05 :> 0.23 :> 0.11 :> (-0.17) :> 0.29 :> 0.00 :> 0.15 :> C.Nil)
            :> ((-0.12) :> 0.14 :> 0.25 :> 0.09 :> 0.18 :> (-0.08) :> 0.22 :> C.Nil)
            :> C.Nil
          w5to3 :: C.Vec 3 (C.Vec 5 Weight)
          w5to3 =
               ( 0.34 :> (-0.16) :> 0.21 :> 0.00 :> 0.18 :> C.Nil)
            :> ( 0.09 :> 0.28 :> (-0.14) :> 0.24 :> 0.11 :> C.Nil)
            :> ((-0.10) :> 0.17 :> 0.26 :> 0.13 :> 0.20 :> C.Nil)
            :> C.Nil
          params = LIFParams 0.8 0.92 0.0
          catNet = snnNetwork3Layer w12to7 w7to5 w5to3 params
          fltNet = flatNetwork3Layer w12to7 w7to5 w5to3 params
          rates12 = [0.07, 0.18, 0.29, 0.41, 0.13, 0.34, 0.22, 0.09, 0.37, 0.16, 0.28, 0.45 :: Double]
          channelTrains =
            [ bernoulliSpikeTrain 120 rate seed'
            | (rate, seed') <- zip rates12 [110 .. 121]
            ]
          inputs =
            [ C.map
                (\i -> (channelTrains P.!! P.fromIntegral i) P.!! t)
                (C.indicesI :: C.Vec 12 (C.Index 12))
            | t <- [0 .. 119]
            ]
      assertEqual "Three-layer irregular exact match" (simulate catNet inputs) (simulate fltNet inputs)
  ]

-- ============================================================
-- 6. Edge Cases
-- ============================================================

edgeTests :: [TestTree]
edgeTests =
  [ testCase "Empty input produces empty output" $ do
      let neuron = lifNeuronCustom defaultLIFParams
      assertEqual "Empty output" [] (simulate neuron ([] :: [Potential]))

  , testCase "Single timestep works" $ do
      let neuron = lifNeuronCustom defaultLIFParams
      assertEqual "Single output" 1 (length (simulate neuron [2.0 :: Potential]))

  , testCase "All-false spikes produce no synaptic current" $ do
      let syn = synapse 100.0
          outputs = simulate syn (P.replicate 10 False)
      assertBool "All zeros" (all (P.== 0.0) outputs)

  , testCase "Bernoulli spike train has correct length" $ do
      assertEqual "Length matches" 100 (length (bernoulliSpikeTrain 100 0.5 42))

  , testCase "Periodic spike train has correct pattern" $ do
      let spikes = periodicSpikeTrain 10 3
      assertEqual "Period 3 spikes" [0, 3, 6, 9] (spikeTimestamps spikes)
  ]

-- ============================================================
-- 7. Output Consistency
-- ============================================================

consistencyTests :: [TestTree]
consistencyTests =
  [ testCase "LIF output length matches input" $ do
      let neuron = lifNeuronCustom defaultLIFParams
          outputs = simulate neuron (constantInput 100 0.3)
      assertEqual "100 outputs" 100 (length outputs)

  , testCase "Layer output count matches timesteps" $ do
      let weights :: C.Vec 3 (C.Vec 4 Weight)
          weights = (0.5 :> 0.3 :> 0.2 :> 0.4 :> C.Nil)
                 :> (0.1 :> 0.6 :> 0.2 :> 0.1 :> C.Nil)
                 :> (0.3 :> 0.2 :> 0.5 :> 0.3 :> C.Nil)
                 :> C.Nil
          layer = snnLayer weights defaultLIFParams
          inputs = P.replicate 10 (True :> False :> True :> False :> C.Nil)
      assertEqual "10 timesteps" 10 (length (simulate layer inputs))

  , testCase "Deterministic: same seed gives same result" $ do
      let chain = synapse 0.5 `composeMorph` lifNeuronCustom defaultLIFParams
          inputs = bernoulliSpikeTrain 100 0.3 42
      assertEqual "Deterministic" (simulate chain inputs) (simulate chain inputs)

  , testCase "Different seeds give different results" $ do
      assertBool "Different seeds differ"
        (bernoulliSpikeTrain 100 0.3 42 P./= bernoulliSpikeTrain 100 0.3 99)
  ]
