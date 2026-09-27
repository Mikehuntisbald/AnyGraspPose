# Read-preference supervision option: implemented, not trained

V74used GT correspondence existence as its gate label. Its60deg/heavy real-hidden dense read gate stays on82.9%of target pixels while dense flow EPE26.787px barely improves zero-flow26.894px. Availability is not actual read quality.

The optional `gate_target=better_than_fallback` labels the read as preferable only when its detached canonical error plus0.005diameter is lower than the fallback error and lookup mass is valid. This is a relative branch preference, not an absolute correctness guarantee. Both errors use teacher geometry only in the loss. No teacher enters inference. Flow and warped-XYZ losses retain fixed GT-supported masks and do not disappear when the predicted gate closes.

The default remains `supported`, so existing V74checkpoints/configs retain their semantics. No preference-supervised checkpoint has been trained or promoted. Large-error correspondence learning also remains necessary; gate rejection alone is not the recovery objective.

Regression result:40passed in8.07s with the same CUBLAS_WORKSPACE_CONFIG=:4096:8 used by training. The first ad-hoc command omitted that environment setting and failed an existing deterministic CUDA bilinear test (38passed,1failed); rerunning with the correct environment and an additional gate/flow gradient test passed all40. Tests show a theoretically supported but actually worse read receives a closing-gate gradient while displacement still receives correction gradients.
