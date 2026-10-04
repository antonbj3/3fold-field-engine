# Cellwise exact rational rule replay

`replay_rule` verifies inequalities between declared quadratic surrogate
scores on a parameter box. Whole-box preference also needs branch existence
for both designs at every point. Each component score is sqrt(max(0,q)), with
q=theta_hat'Q theta_hat and interval matrix entries interpreted exactly.
For the selected score, all components must be bounded. For the other score,
an anchor must satisfy every declared sign guard. Every non-excluded branch
pair must be present; an empty family, omitted component or duplicate pair
returns UNKNOWN. All decisions are recomputed; producer counters are ignored.

Centre-radius interval evaluation covers interior extrema. Corners alone are
not a quadratic-extremum license. For margin eta and selected norm upper u,
the checker proves both q_other-q_selected>2 eta u+eta² and
q_other>2 eta u+eta², so clamping a negative selected quadratic cannot create
a false preference. Exact witnesses are checked against recomputed bounds.

`exact_partition` verifies a recursive guillotine tiling of closed positive-
volume boxes, including common boundaries and excluding gaps/interior overlap.
It intentionally abstains for partitions outside that representation. Summed
volume alone does not prove coverage. Dimension is bounded at eight.

Placement: field experimental verification beside verified-window operations.
CERTIFIED_SURROGATE_RULE certifies the stated coefficient family and branch
guards. Linking upper surrogates/anchors to a physical objective, FE extraction
error, measurement, source/model hashes and later query routing remains the
consumer's separate contract. A preference does not prove design feasibility.

Branch coverage is a separate mathematical obligation. For each cell, an
unconditionally active branch of each design (every affine guard's exact upper
bound <= 0) is a sufficient cover. Only when every cell has that verified cover
does the result say CERTIFIED_SURROGATE_RULE with domain_coverage=CERTIFIED.
Otherwise CERTIFIED_BRANCH_CONDITIONAL_RULE certifies the inequalities whenever
both designs have active branches, with domain_coverage=UNKNOWN. It does not
license routing on the entire parameter box or prove equilibrium existence.
covered_cells counts cells with the sufficient cover; union covers outside this
representation may be valid but need a separate proof.

For example, x>=3/4 for one design and x<=1/4 for the other leaves no common
active point in [0,1], even if every stored component inequality passes. Cell
tiling and listing potential masks do not establish branch-domain coverage.
