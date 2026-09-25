# V27: read actual JEPA outputs, without ideal geometry

Completed; no readout promoted. This is a bounded training-data diagnostic,
not native validation or evidence that the latent contains no pose information.

The frozen source is V21 step1000, SHA256
`c353d5bda97ff33782c65b5adde8e492303ef6ea30e36c30f514e4e6b1d77867`.
256 observation groups contain four estimated poses each: correct, positive10°,
negative10°, and mixed error. All four use exactly the same sensor RGB-D crop,
constructed from the mixed estimate. GT defines supervised perturbations and
labels; it does not supply geometry or features to the readout. Physical-sequence
hash partitioning yields192 train groups and64 heldout groups (not native val).
Half the groups have added real-texture RGB-D occlusion.

Three fresh readouts train1000 updates, seed42, batch32, LR3e-4, same objective
and geometry statistics. All replace the legacy residual-magnitude gate with
the same evidence gate. Appearance is either legacy observed-visible features,
fully decoded features, or final JEPA patch latent. The last has a256→128
projection rather than768→128 and thus fewer parameters. Raw-patch access is
diagnostic only; it has not been installed as a production pose branch.

|Readout|Heldout non-symmetric rotation|Added-occlusion rotation|Zero-base drift|
|---|---:|---:|---:|
|Existing V21 head, same cases|9.025°|9.341°|2.089°|
|Fresh observed-visible|10.003°|10.067°|4.225°|
|Fresh decoded|9.771°|9.737°|1.896°|
|Fresh patch|9.439°|9.707°|3.090°|

Rotation starts at10°. The non-symmetric comparison has90 cases from38 physical
sequences; the augmented subset has48 cases. Fresh heads improve their training
rotation error, but none beats the existing head on heldout rotation. Merely
opening access to the actual patch latent did not solve pose in this probe.
This supports testing input/reconstruction fidelity rather than another oracle
readout calibration; it does not prove a universal inability to decode pose.

Cached local17D/global24D relation statistics reproduce the actual readout.
Tests check equivalence and independence of labels. No teacher forward, GT
geometry input, backbone update, native validation, or official test occurs.
Runtime: `/tmp/dexycb_actual_readout_v27`; remote artifact suffix:
`unified_jepa_20260921/actual_readout_v27`. Receipts, readout checkpoints, source
archive and cache hashes are also copied into the local report directory.
