# V53 formal JEPA recovery monitoring

Full-state continuation of V52 balanced200;5000 additional updates. Same loss/architecture/data; CE weight0.1. Pose frozen, history and DINO feature losses off.

Step numbers continue the V52 clock:200 is the inherited source,700/1200/2700/5200 correspond to500/1000/2500/5000 new updates. Both sets are previously inspected training-partition physical holdouts; use as development monitoring, not an unseen test.

## usual: heavy subset

|Step|Real CAD XYZ mm|Real depth mm|Proxy CAD XYZ mm|Proxy depth mm|Real surface angle deg|Proxy surface angle deg|
|---|---:|---:|---:|---:|---:|---:|
|step200|12.278|10.424|16.566|13.026|26.845|24.436|

## confirmation: heavy subset

|Step|Real CAD XYZ mm|Real depth mm|Proxy CAD XYZ mm|Proxy depth mm|Real surface angle deg|Proxy surface angle deg|
|---|---:|---:|---:|---:|---:|---:|
|step200|11.604|11.859|11.725|9.653|28.713|21.446|

Original real depth and CAD-proxy labels/masks are unchanged. Canonical XYZ measures CAD identity; camera XYZ is depth lifted through rays. Every50 steps saves a full checkpoint; evaluated milestones retain immutable full checkpoints. Automatic stops are runtime/nonfinite failures or the budget boundary; metric changes are reported without selecting or replacing the default model.
