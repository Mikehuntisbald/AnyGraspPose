# V72 direct-distance search trial stopped for performance

Candidate stopped deliberately after14updates; checkpoints and all rank logs remain remote. Direct non-GEMM cdist over dense queries and8192CAD points took about8–9seconds per warm step despite roughly0.12seconds of data loading in some steps. It is not a completed accuracy comparison.

The fast replacement uses GEMM to shortlist32neighbors then direct distances to rerank8. A CUDA benchmark at8x2048x8192reduced median search0.28449→0.00587seconds with identical sampled indices/coordinates. Near-point CPU and adversarial decoder tests passed. This synthetic equality is not a blanket bitwise guarantee on all real inputs. A separate immutable fast run restarts from the original source, retaining the matched200update budget. The stopped trial is preserved, not resumed under silently modified source.
