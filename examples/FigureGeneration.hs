-- | Extended experiments for publication-quality results.
--
-- Generates CSV data for:
-- 5. Scalability: varying network sizes (8→4, 16→8→4, etc.)
-- 6. Extended categorical law verification (tensor, braiding, interchange)
-- 7. Tensor product (parallel composition) demonstration
-- 8. Robustness across random weight configurations
-- 9. Compositional reuse demonstration
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

-- | Base directory for output
baseDir :: String
baseDir = "../paper/data/"

-- | Simulation parameters
numTimesteps :: P.Int
numTimesteps = 200

seed :: P.Int
seed = 42

-- ============================================================
-- Helpers
-- ============================================================

sfToDouble :: Potential -> P.Double
sfToDouble x =
  let bv = pack x :: BitVector 32
      sv = unpack bv :: Signed 32
  in  P.fromIntegral sv P./ (2 P.^ (16 :: P.Int))

boolToStr :: P.Bool -> String
boolToStr P.True  = "1"
boolToStr P.False = "0"

dblToStr :: P.Double -> String
dblToStr = P.show

intToStr :: P.Int -> String
intToStr = P.show

-- ============================================================
-- Main
-- ============================================================

main :: IO ()
main = do
  putStrLn "=== Extended Experiments for Publication ==="
  putStrLn ""

  experiment5_scalability
  experiment6_extendedLaws
  experiment7_tensorProduct
  experiment8_robustness
  experiment9_reuse

  putStrLn "=== All extended experiments complete ==="

-- ============================================================
-- Experiment 5: Scalability
-- ============================================================

experiment5_scalability :: IO ()
experiment5_scalability = do
  putStrLn "--- Experiment 5: Scalability Sweep ---"

  let params = defaultLIFParams

  -- Baseline scale: 4 -> 3
  let w_4_3 :: C.Vec 3 (C.Vec 4 Weight)
      w_4_3 = ( 0.5 :>  0.3 :> (-0.2) :>  0.4 :> C.Nil)
           :> ( 0.1 :>  0.6 :>  0.2 :> (-0.1) :> C.Nil)
           :> ((-0.3) :>  0.2 :>  0.5 :>  0.3 :> C.Nil)
           :> C.Nil

      w_3_2 :: C.Vec 2 (C.Vec 3 Weight)
      w_3_2 = ( 0.4 :> (-0.2) :>  0.6 :> C.Nil)
           :> ( 0.3 :>  0.5 :> (-0.1) :> C.Nil)
           :> C.Nil

      inputs4 :: [C.Vec 4 Spike]
      inputs4 = take numTimesteps
        [ s0 :> s1 :> s2 :> s3 :> C.Nil
        | (s0, (s1, (s2, s3))) <-
            zip (bernoulliSpikeTrain numTimesteps 0.3 42)
                (zip (bernoulliSpikeTrain numTimesteps 0.2 43)
                     (zip (bernoulliSpikeTrain numTimesteps 0.4 44)
                          (bernoulliSpikeTrain numTimesteps 0.1 45)))
        ]

      catL_4_3 = snnLayer w_4_3 params
      fltL_4_3 = flatLayer w_4_3 params
      catOut_4_3 = simulate catL_4_3 inputs4
      fltOut_4_3 = simulate fltL_4_3 inputs4
      match_4_3 = P.all P.id (zipWith (P.==) catOut_4_3 fltOut_4_3)
      rates_4_3 = map (\i -> firingRate (map (\v -> v C.!! i) catOut_4_3))
                      [0, 1, 2 :: C.Index 3]

      catNet_4_3_2 = snnNetwork2Layer w_4_3 w_3_2 params
      fltNet_4_3_2 = flatNetwork2Layer w_4_3 w_3_2 params
      catOut_4_3_2 = simulate catNet_4_3_2 inputs4
      fltOut_4_3_2 = simulate fltNet_4_3_2 inputs4
      match_4_3_2 = P.all P.id (zipWith (P.==) catOut_4_3_2 fltOut_4_3_2)
      rates_4_3_2 = map (\i -> firingRate (map (\v -> v C.!! i) catOut_4_3_2))
                        [0, 1 :: C.Index 2]

  -- Scale 1: 8 -> 4
  let w_8_4 :: C.Vec 4 (C.Vec 8 Weight)
      w_8_4 = ( 0.5 :>  0.3 :> (-0.2) :>  0.4 :>  0.1 :> (-0.3) :>  0.2 :>  0.6 :> C.Nil)
           :> ( 0.1 :>  0.6 :>  0.2 :> (-0.1) :>  0.4 :>  0.2 :> (-0.5) :>  0.3 :> C.Nil)
           :> ((-0.3) :>  0.2 :>  0.5 :>  0.3 :> (-0.1) :>  0.4 :>  0.3 :> (-0.2) :> C.Nil)
           :> ( 0.4 :> (-0.2) :>  0.6 :>  0.1 :>  0.3 :> (-0.4) :>  0.5 :>  0.2 :> C.Nil)
           :> C.Nil

      catL_8_4 = snnLayer w_8_4 params
      fltL_8_4 = flatLayer w_8_4 params

      inputs8 :: [C.Vec 8 Spike]
      inputs8 = take numTimesteps
        [ s0 :> s1 :> s2 :> s3 :> s4 :> s5 :> s6 :> s7 :> C.Nil
        | (s0,(s1,(s2,(s3,(s4,(s5,(s6,s7))))))) <-
            zip (bernoulliSpikeTrain numTimesteps 0.3 42)
                (zip (bernoulliSpikeTrain numTimesteps 0.2 43)
                     (zip (bernoulliSpikeTrain numTimesteps 0.4 44)
                          (zip (bernoulliSpikeTrain numTimesteps 0.1 45)
                               (zip (bernoulliSpikeTrain numTimesteps 0.35 46)
                                    (zip (bernoulliSpikeTrain numTimesteps 0.25 47)
                                         (zip (bernoulliSpikeTrain numTimesteps 0.15 48)
                                              (bernoulliSpikeTrain numTimesteps 0.3 49)))))))
        ]

      catOut_8_4 = simulate catL_8_4 inputs8
      fltOut_8_4 = simulate fltL_8_4 inputs8
      match_8_4 = P.all P.id (zipWith (P.==) catOut_8_4 fltOut_8_4)

  -- Scale 2: 16 -> 8 -> 4  (two layers)
  let w_16_8 :: C.Vec 8 (C.Vec 16 Weight)
      w_16_8 = C.repeat (C.repeat 0.125)  -- uniform small weights

      w2_8_4 :: C.Vec 4 (C.Vec 8 Weight)
      w2_8_4 = w_8_4  -- reuse

      catNet_16_8_4 = snnLayer w_16_8 params `composeMorph` snnLayer w2_8_4 params
      fltNet_16_8_4 = flatNetwork2Layer w_16_8 w2_8_4 params

      inputs16 :: [C.Vec 16 Spike]
      inputs16 = take numTimesteps
        [ s0:>s1:>s2:>s3:>s4:>s5:>s6:>s7:>s8:>s9:>sa:>sb:>sc:>sd:>se:>sf:>C.Nil
        | (s0,(s1,(s2,(s3,(s4,(s5,(s6,(s7,(s8,(s9,(sa,(sb,(sc,(sd,(se,sf))))))))))))))) <-
            zip (bernoulliSpikeTrain numTimesteps 0.3 50)
                (zip (bernoulliSpikeTrain numTimesteps 0.2 51)
                     (zip (bernoulliSpikeTrain numTimesteps 0.4 52)
                          (zip (bernoulliSpikeTrain numTimesteps 0.1 53)
                               (zip (bernoulliSpikeTrain numTimesteps 0.35 54)
                                    (zip (bernoulliSpikeTrain numTimesteps 0.25 55)
                                         (zip (bernoulliSpikeTrain numTimesteps 0.15 56)
                                              (zip (bernoulliSpikeTrain numTimesteps 0.3 57)
                                                   (zip (bernoulliSpikeTrain numTimesteps 0.2 58)
                                                        (zip (bernoulliSpikeTrain numTimesteps 0.4 59)
                                                             (zip (bernoulliSpikeTrain numTimesteps 0.1 60)
                                                                  (zip (bernoulliSpikeTrain numTimesteps 0.35 61)
                                                                       (zip (bernoulliSpikeTrain numTimesteps 0.25 62)
                                                                            (zip (bernoulliSpikeTrain numTimesteps 0.15 63)
                                                                                 (zip (bernoulliSpikeTrain numTimesteps 0.3 64)
                                                                                      (bernoulliSpikeTrain numTimesteps 0.2 65)))))))))))))))
        ]

      catOut_16 = simulate catNet_16_8_4 inputs16
      fltOut_16 = simulate fltNet_16_8_4 inputs16
      match_16 = P.all P.id (zipWith (P.==) catOut_16 fltOut_16)

  -- Compute firing rates for each scale
  let rates_8_4 = map (\i -> firingRate (map (\v -> v C.!! i) catOut_8_4))
                      [0, 1, 2, 3 :: C.Index 4]
      rates_16 = map (\i -> firingRate (map (\v -> v C.!! i) catOut_16))
                     [0, 1, 2, 3 :: C.Index 4]

  let formatRates width rates =
        L.intercalate "," (map dblToStr rates ++ replicate (width P.- length rates) "NaN")

  let header = "config,layers,input_width,output_width,exact_match,out0_rate,out1_rate,out2_rate,out3_rate"
      rows = [ "4to3,1,4,3," ++ P.show match_4_3 ++ "," ++ formatRates 4 rates_4_3
             , "4to3to2,2,4,2," ++ P.show match_4_3_2 ++ "," ++ formatRates 4 rates_4_3_2
             , "8to4,1,8,4," ++ P.show match_8_4 ++ "," ++ formatRates 4 rates_8_4
             , "16to8to4,2,16,4," ++ P.show match_16 ++ "," ++ formatRates 4 rates_16
             ]

  writeFile (baseDir ++ "experiment5_scalability.csv") (header ++ "\n" ++ unlines rows)

  putStrLn ("  8->4 single layer match: " ++ P.show match_8_4)
  putStrLn ("  16->8->4 two-layer match: " ++ P.show match_16)
  putStrLn ("  Firing rates 8->4: " ++ P.show rates_8_4)
  putStrLn ("  Firing rates 16->8->4: " ++ P.show rates_16)
  putStrLn ""

-- ============================================================
-- Experiment 6: Extended Law Verification
-- ============================================================

experiment6_extendedLaws :: IO ()
experiment6_extendedLaws = do
  putStrLn "--- Experiment 6: Extended Categorical Laws ---"

  let params = defaultLIFParams
      neuron = lifNeuronCustom params
      syn05 = synapse 0.5
      syn03 = synapse 0.3
      syn08 = synapse 0.8

      -- Test inputs
      spikeIn = bernoulliSpikeTrain 100 0.3 seed
      potIn = simulate syn05 spikeIn

  -- 1. Left identity: id >>> f = f
  let idLeft = idMorph `composeMorph` neuron
      out_idL = simulate idLeft potIn
      out_dir = simulate neuron potIn
      identityLeft = out_idL P.== out_dir

  -- 2. Right identity: f >>> id = f
  let idRight = neuron `composeMorph` idMorph
      out_idR = simulate idRight potIn
      identityRight = out_idR P.== out_dir

  -- 3. Associativity: (f >>> g) >>> h = f >>> (g >>> h)
  let lAssoc = (syn05 `composeMorph` neuron) `composeMorph` syn08
      rAssoc = syn05 `composeMorph` (neuron `composeMorph` syn08)
      out_LA = simulate lAssoc spikeIn
      out_RA = simulate rAssoc spikeIn
      associativity = out_LA P.== out_RA

  -- 4. Tensor independence: (f ⊗ g)(a,c) = (f(a), g(c))
  let fg = tensorMorph syn05 syn03
      tensorInputs = zip spikeIn (bernoulliSpikeTrain 100 0.4 99)
      out_tensor = simulate fg tensorInputs
      out_f_only = simulate syn05 spikeIn
      out_g_only = simulate syn03 (bernoulliSpikeTrain 100 0.4 99)
      tensorIndep = P.all (\((a,b),(c,d)) -> a P.== c P.&& b P.== d)
                    (zip out_tensor (zip out_f_only out_g_only))

  -- 5. Braiding involution: braid >>> braid = id
  let braidBraid = braidMorph `composeMorph` braidMorph
      pairInputs = zip potIn (constantInput 100 0.5) :: [(Potential, Potential)]
      out_bb = simulate braidBraid pairInputs
      braidInvolution = out_bb P.== pairInputs

  -- 6. Interchange law: (f1 ⊗ g1) >>> (f2 ⊗ g2) = (f1>>>f2) ⊗ (g1>>>g2)
  let f1 = syn05
      g1 = syn03
      f2 = neuron
      g2 = lifNeuronCustom (LIFParams 0.8 0.85 0.0)
      lhs = tensorMorph f1 g1 `composeMorph` tensorMorph f2 g2
      rhs = tensorMorph (f1 `composeMorph` f2) (g1 `composeMorph` g2)
      ichInputs = zip spikeIn (bernoulliSpikeTrain 100 0.4 99) :: [(Spike, Spike)]
      out_lhs = simulate lhs ichInputs
      out_rhs = simulate rhs ichInputs
      interchange = out_lhs P.== out_rhs

  let header = "law,holds"
      rows = [ "identity_left," ++ P.show identityLeft
             , "identity_right," ++ P.show identityRight
             , "associativity," ++ P.show associativity
             , "tensor_independence," ++ P.show tensorIndep
             , "braiding_involution," ++ P.show braidInvolution
             , "interchange," ++ P.show interchange
             ]

  writeFile (baseDir ++ "experiment6_extended_laws.csv") (header ++ "\n" ++ unlines rows)

  putStrLn ("  Identity left:       " ++ P.show identityLeft)
  putStrLn ("  Identity right:      " ++ P.show identityRight)
  putStrLn ("  Associativity:       " ++ P.show associativity)
  putStrLn ("  Tensor independence: " ++ P.show tensorIndep)
  putStrLn ("  Braiding involution: " ++ P.show braidInvolution)
  putStrLn ("  Interchange law:     " ++ P.show interchange)
  putStrLn ""

-- ============================================================
-- Experiment 7: Tensor Product (Parallel Composition)
-- ============================================================

experiment7_tensorProduct :: IO ()
experiment7_tensorProduct = do
  putStrLn "--- Experiment 7: Tensor Product Demonstration ---"

  let params = defaultLIFParams

  -- Two independent pathways
  -- Pathway A: spike -> synapse(0.5) -> LIF neuron
  -- Pathway B: spike -> synapse(0.3) -> LIF neuron
  let pathA = synapse 0.5 `composeMorph` lifNeuronCustom params
      pathB = synapse 0.3 `composeMorph` lifNeuronCustom params

      -- Parallel: (pathA ⊗ pathB)
      parallel = tensorMorph pathA pathB

      spikeA = bernoulliSpikeTrain numTimesteps 0.3 42
      spikeB = bernoulliSpikeTrain numTimesteps 0.5 99
      inputs = zip spikeA spikeB :: [(Spike, Spike)]

      parOutputs = simulate parallel inputs
      outA = map P.fst parOutputs
      outB = map P.snd parOutputs

      -- Independent runs for verification
      indepA = simulate pathA spikeA
      indepB = simulate pathB spikeB

      matchA = outA P.== indepA
      matchB = outB P.== indepB

  let header = "timestep,input_a,input_b,tensor_out_a,tensor_out_b,indep_out_a,indep_out_b,match_a,match_b"
      rows = [ L.intercalate ","
               [ intToStr t
               , boolToStr ia, boolToStr ib
               , boolToStr ta, boolToStr tb
               , boolToStr da, boolToStr db
               , boolToStr (ta P.== da), boolToStr (tb P.== db)
               ]
             | (t, (ia, ib, ta, tb, da, db)) <-
                 zip [0..] (zip6 spikeA spikeB outA outB indepA indepB)
             ]

  writeFile (baseDir ++ "experiment7_tensor.csv") (header ++ "\n" ++ unlines rows)

  putStrLn ("  Tensor pathway A matches independent: " ++ P.show matchA)
  putStrLn ("  Tensor pathway B matches independent: " ++ P.show matchB)
  putStrLn ("  Pathway A firing rate: " ++ dblToStr (firingRate outA))
  putStrLn ("  Pathway B firing rate: " ++ dblToStr (firingRate outB))
  putStrLn ""

  where
    zip6 (a:as) (b:bs) (c:cs) (d:ds) (e:es) (f:fs) =
      (a,b,c,d,e,f) : zip6 as bs cs ds es fs
    zip6 _ _ _ _ _ _ = []

-- ============================================================
-- Experiment 8: Robustness Across Weight Configurations
-- ============================================================

experiment8_robustness :: IO ()
experiment8_robustness = do
  putStrLn "--- Experiment 8: Robustness (Multiple Weight Configs) ---"

  let params = defaultLIFParams

      -- Generate 5 different 4->3 weight matrices using different seed offsets
      mkWeights :: Weight -> Weight -> Weight -> Weight ->
                   Weight -> Weight -> Weight -> Weight ->
                   Weight -> Weight -> Weight -> Weight ->
                   C.Vec 3 (C.Vec 4 Weight)
      mkWeights a b c d e f g h i j k l =
             (a :> b :> c :> d :> C.Nil)
          :> (e :> f :> g :> h :> C.Nil)
          :> (i :> j :> k :> l :> C.Nil)
          :> C.Nil

      -- Config 1: positive weights
      w1 = mkWeights 0.5 0.3 0.2 0.4  0.1 0.6 0.2 0.1  0.3 0.2 0.5 0.3

      -- Config 2: mixed positive/negative
      w2 = mkWeights 0.5 0.3 (-0.2) 0.4  0.1 0.6 0.2 (-0.1)  (-0.3) 0.2 0.5 0.3

      -- Config 3: large weights
      w3 = mkWeights 1.5 (-0.8) 1.2 (-0.5)  0.9 1.1 (-0.7) 0.6  (-1.0) 0.4 0.8 1.3

      -- Config 4: small weights
      w4 = mkWeights 0.05 0.03 0.02 0.04  0.01 0.06 0.02 0.01  0.03 0.02 0.05 0.03

      -- Config 5: sparse (many zeros)
      w5 = mkWeights 0.5 0.0 0.0 0.0  0.0 0.6 0.0 0.0  0.0 0.0 0.5 0.0

      configs = [("positive", w1), ("mixed", w2), ("large", w3),
                 ("small", w4), ("sparse", w5)]

      -- Shared inputs
      inputs :: [C.Vec 4 Spike]
      inputs = take numTimesteps
        [ s0 :> s1 :> s2 :> s3 :> C.Nil
        | (s0, (s1, (s2, s3))) <-
            zip (bernoulliSpikeTrain numTimesteps 0.3 42)
                (zip (bernoulliSpikeTrain numTimesteps 0.2 43)
                     (zip (bernoulliSpikeTrain numTimesteps 0.4 44)
                          (bernoulliSpikeTrain numTimesteps 0.1 45)))
        ]

      testConfig (name, w) =
        let catL = snnLayer w params
            fltL = flatLayer w params
            catOut = simulate catL inputs
            fltOut = simulate fltL inputs
            match = P.all P.id (zipWith (P.==) catOut fltOut)
            rates = map (\i -> firingRate (map (\v -> v C.!! i) catOut))
                        [0, 1, 2 :: C.Index 3]
        in  (name, match, rates)

      results = map testConfig configs

  let header = "config,exact_match,rate_out0,rate_out1,rate_out2"
      rows = [ name ++ "," ++ P.show match ++ "," ++ L.intercalate "," (map dblToStr rates)
             | (name, match, rates) <- results
             ]

  writeFile (baseDir ++ "experiment8_robustness.csv") (header ++ "\n" ++ unlines rows)

  P.mapM_ (\(name, match, rates) ->
    putStrLn ("  " ++ name ++ ": match=" ++ P.show match ++
              " rates=" ++ P.show rates)) results
  putStrLn ""

-- ============================================================
-- Experiment 9: Compositional Reuse
-- ============================================================

experiment9_reuse :: IO ()
experiment9_reuse = do
  putStrLn "--- Experiment 9: Compositional Reuse ---"

  let params = defaultLIFParams

      -- Define 5 primitive morphisms
      syn05 = synapse 0.5     -- Spike -> Potential
      syn03 = synapse 0.3     -- Spike -> Potential
      lif   = lifNeuronCustom params  -- Potential -> Spike
      wire  = wireSynapse      -- Spike -> Potential (weight=1)
      thr   = thresholdNeuron 0.5     -- Potential -> Spike

      -- Compose 4 distinct architectures from these primitives

      -- Arch 1: syn(0.5) >>> LIF (basic sensory neuron)
      arch1 = syn05 `composeMorph` lif

      -- Arch 2: syn(0.3) >>> LIF (weak coupling)
      arch2 = syn03 `composeMorph` lif

      -- Arch 3: wire >>> threshold (instant relay)
      arch3 = wire `composeMorph` thr

      -- Arch 4: syn(0.5) >>> LIF >>> wire >>> LIF (two-stage integration)
      arch4 = syn05 `composeMorph` lif `composeMorph` wire `composeMorph` lif

      -- Test each architecture
      spikeIn = bernoulliSpikeTrain numTimesteps 0.3 42

      out1 = simulate arch1 spikeIn
      out2 = simulate arch2 spikeIn
      out3 = simulate arch3 spikeIn
      out4 = simulate arch4 spikeIn

      rate1 = firingRate out1
      rate2 = firingRate out2
      rate3 = firingRate out3
      rate4 = firingRate out4

  let header = "architecture,primitives_used,composition_ops,firing_rate,spike_count"
      rows = [ "sensory_neuron,syn05+lif,1," ++ dblToStr rate1 ++ "," ++ intToStr (spikeCount out1)
             , "weak_coupling,syn03+lif,1," ++ dblToStr rate2 ++ "," ++ intToStr (spikeCount out2)
             , "instant_relay,wire+thr,1," ++ dblToStr rate3 ++ "," ++ intToStr (spikeCount out3)
             , "two_stage,syn05+lif+wire+lif,3," ++ dblToStr rate4 ++ "," ++ intToStr (spikeCount out4)
             ]

  writeFile (baseDir ++ "experiment9_reuse.csv") (header ++ "\n" ++ unlines rows)

  putStrLn ("  Arch 1 (sensory neuron): rate=" ++ dblToStr rate1)
  putStrLn ("  Arch 2 (weak coupling):  rate=" ++ dblToStr rate2)
  putStrLn ("  Arch 3 (instant relay):  rate=" ++ dblToStr rate3)
  putStrLn ("  Arch 4 (two-stage):      rate=" ++ dblToStr rate4)
  putStrLn ""
