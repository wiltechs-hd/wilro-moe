"""Single source of truth for LIBERO instruction rewriting.

Some LIBERO tasks fail not on control but on GROUNDING: the frozen Qwen3-VL
encoder can't bind a product name ("alphabet soup") to the right prop when a
visually-similar object shares the scene. Rephrasing the reference with a
visual primitive the VLM CAN ground (color + shape: "the blue can of alphabet
soup") gives the policy a handle it can actually use.

DESIGN
------
- Keyed by the canonical instruction STRING, not the task index. The benchmark
  per-suite task_id (0-9) is a *permutation* of the parquet task_index (0-39),
  and training reads strings from tasks.parquet while eval reads them from the
  LIBERO benchmark env. Keying by string makes the rewrite identical on both
  paths regardless of indexing — no permutation table to keep in sync.
- Identity fallback: tasks NOT in REPHRASINGS pass through unchanged, so the
  well-performing tasks need no entries. One exception, and it is deliberate:
  all ten libero_spatial tasks are listed even though nine of them work,
  because they share one scene and must therefore share one set of nouns —
  see the block above those entries.
- ONE function, `rewrite_instruction`, called by training, RL rollout, and eval
  alike. As long as every path runs the raw task string through it, train and
  deploy can never diverge on phrasing.

CHAIN-OF-THOUGHT (CoT) REWRITES
-------------------------------
For tasks where the model mis-grounds the target object (e.g. grasping the
midpoint between two reference objects instead of the actual target), a
flat descriptive rephrase is insufficient — the model needs an explicit
reasoning trace that:

  1. Names the TARGET object and its visual signature (shape/depth/contents).
  2. SELECTS it from the candidates by a spatial relation — "whichever of those
     two is nearer to X", never "at position X". A relation used as a
     coordinate pins a region and the policy will fly to that region; used as
     a selector it picks one object and the final aim comes from vision.
  3. States the ACTION to perform on the resolved target.

Step 2 replaced an earlier "resolve common confusions (not the midpoint, not
the anchor)" step. Negation is weak for a frozen encoder and the phrase inside
it lands anyway: "not the midpoint between the plate and the ramekin" was read
as "midpoint between the plate and the ramekin". Phrase selectors positively.

Because the VLM (Qwen3-VL) is frozen and does NOT autoregressively generate
text at inference time, the CoT cannot be "generated" by the model itself.
Instead, the CoT is PRE-WRITTEN in this table and injected as the language
input the VLM encodes. The VLM's KV cache then carries the structured
grounding trace, which the trainable DiT decoder cross-attends to at every
denoising step. Training on these CoT strings teaches the DiT to *use* the
trace; at eval/RL-rollout time the identical string flows through
`rewrite_instruction`, so the model sees the same reasoning it was trained on
— no runtime generation required.

This is analogous to "pre-filled assistant reasoning" in VLA literature: the
reasoning is provided as context, not produced online.

USAGE
-----
    from task_rewrites import rewrite_instruction
    task = rewrite_instruction(raw_task_string)   # safe on ANY string

Run verify_against_benchmark() once after editing this file to confirm every
key exactly matches a real LIBERO task string (catches typos that would
silently no-op).
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Rephrasings: canonical original string  ->  replacement (may include CoT).
#
# Two value styles are supported (both are plain strings):
#
#   (a) Flat descriptive rephrase  —  original nouns swapped for visual
#       primitives, no reasoning trace. Backwards compatible.
#
#   (b) Chain-of-thought rephrase  —  a multi-clause string that first
#       identifies the target, then locates it, then states the action. Use
#       the helper `cot(...)` (defined below) to keep the format consistent.
#
# The model does NOT differentiate between (a) and (b): both are just the
# language string the VLM encodes. The CoT form is strictly richer and tends
# to help on tasks where flat rephrasing still leaves spatial ambiguity.
#
# Still unattested, so still un-rewritten: orange juice, ketchup, bbq sauce,
# milk, salad dressing, chocolate pudding (libero_object). Nothing in this repo
# records what they look like, and a wrong guess is worse than the canonical
# noun, which at least matches the demo language. Get them from
# `kv_grounding_probe.py --annotate_detect` before adding entries.
# ---------------------------------------------------------------------------

def cot(
    target: str,
    location: str,
    action: str,
    *,
    not_the: str = "",
    visual: str = "",
) -> str:
    """Format a chain-of-thought grounding trace as a single instruction string.

    The output is a compact, declarative trace the frozen VLM can encode and
    the trainable DiT can cross-attend to. It is NOT free-form generation —
    it is a pre-written reasoning template filled per task.

    Args:
        target:   The object to grasp, with visual discriminators
                  (e.g. "the deep speckled bowl", "the blue can of alphabet
                  soup"). A bare category noun is not a target: "the black bowl"
                  names both candidates when two are in frame.
        location: How to FIND the target. Use the spatial relation only to pick
                  WHICH object among look-alikes; never to describe where in the
                  frame to reach. Phrasings that parameterise a position on the
                  anchor axis ("in the gap between A and B", "closer to the A
                  side", "the midpoint of") are read literally by the policy and
                  make it interpolate between the anchors instead of localising
                  the object — see the "between" entry below. Prefer
                  "among the <look-alikes>, the target is the one that <relation>"
                  plus "aim at the center of the <object> itself".
        action:   The action to perform, phrased as an imperative
                  (e.g. "grasp the black bowl and place it on the plate").
        not_the:  Optional anti-grounding cue for the most likely confusion, and
                  only when the confusion is another OBJECT ("not the container
                  itself"). Do not use it against a POSITION: negation is a weak
                  signal for a frozen encoder, so "not the midpoint between A and
                  B" mostly just injects "midpoint between A and B".
        visual:   Optional extra visual signature for the target
                  (e.g. "a bowl with a dark rim and speckled granular contents").

    Returns:
        A single string. For libero_spatial, do not call this directly -- use
        `_spatial(selector)`, which fills every field from the shared vocabulary
        so the ten tasks cannot drift apart. Its output looks like:

        "Target: the deep speckled bowl — a bowl with a dark rim and speckled
        granular contents, clearly deeper than the small shallow empty cup.
        Location: there are two such bowls; the target is whichever of those two
        is nearer to the round white plate with the red rim, and the other one is
        not the target. Aim at the center of the target bowl itself. Action:
        grasp that bowl and place it on the round white plate with the red rim."
    """
    parts = [f"Target: {target}"]
    if visual:
        parts.append(f" — {visual}")
    parts.append(f". Location: {location}")
    if not_the:
        parts.append(f"; {not_the}")
    parts.append(f". Action: {action}.")
    return "".join(parts)


# ---------------------------------------------------------------------------
# libero_spatial shared vocabulary
# ---------------------------------------------------------------------------
# All ten libero_spatial tasks share ONE scene and ONE object set, so they must
# share ONE set of nouns. Before this was factored out, the same ramekin was
# "the small shallow empty cup" in the between task and "the small round silver
# container" in the two ramekin tasks, and a bare "black bowl" in the six that
# had no rewrite at all -- three mutually inconsistent descriptions, two of them
# perceptually contradictory, for objects sitting side by side in one frame.
# Constants make that drift structurally impossible rather than a thing to
# remember.
#
# The wording is the one qwen_color_probe.py validated (see the block below):
# no colour words, no "ramekin", containers separated by depth/size/emptiness
# and contents, which is how Qwen itself described them.
_SP_BOWL = "the deep speckled bowl"
_SP_BOWLS = "there are two such bowls"
_SP_PLATE = "the round white plate with the red rim"
_SP_CUP = "the small shallow empty cup"
_SP_VISUAL = ("a bowl with a dark rim and speckled granular contents, clearly "
              "deeper than the small shallow empty cup")


def _spatial(selector: str) -> str:
    """Build one libero_spatial rewrite from the shared vocabulary.

    `selector` completes "the target is whichever of those two ..." and is the
    ONLY thing that varies between tasks -- each keeps its own canonical
    relation, because those were measured and nine of ten separate the two bowls
    by >=2x. Phrase it positively: negation is weak for a frozen encoder, and
    the old "not the midpoint between the plate and the ramekin" only succeeded
    in injecting "midpoint between the plate and the ramekin".

    Kept terse on purpose: _lang_max_len is 128 and the longest of these renders
    to ~105 tokens. `visual` already spells out the contents and the depth
    contrast, so `target` and the location opener must not repeat them -- an
    earlier draft said "speckled granular contents" three times in one string
    and cost ~10 tokens of headroom for nothing. The real count is printed every
    run by _report_lang_budget; if it ever says TRUNCATED, shorten here.
    """
    return cot(
        target=_SP_BOWL,
        location=f"{_SP_BOWLS}; the target is whichever of those two {selector}. "
                 f"Aim at the center of the target bowl itself",
        action=f"grasp that bowl and place it on {_SP_PLATE}",
        visual=_SP_VISUAL,
    )


REPHRASINGS: dict[str, str] = {
    # ---- libero_10 (long) — object-identity grounding ----
    # T0: two round cans, distinguished only by color (red vs blue).
    "put both the alphabet soup and the tomato sauce in the basket":
        "put both the blue can of alphabet soup and the red can of tomato sauce in the basket",
    # T7: a can + a box — shape already separates them; color is a cheap bonus.
    "put both the alphabet soup and the cream cheese box in the basket":
        "put both the blue can of alphabet soup and the cream cheese box in the basket",
    # T1: PENDING cream-cheese/butter colors — uncomment once known.
    "put both the cream cheese box and the butter in the basket":
         "put both the silver purple cream cheese box and the red butter box in the basket",

    # ---- libero_spatial ----
    # The failing task grounds the SPATIAL RELATION (between/next-to/on) to the
    # GEOMETRIC MIDPOINT of the anchors rather than to the object that satisfies
    # the relation. A flat rephrase ("between the plate and the ramekin, closer
    # to the plate") still leaves the midpoint as a strong attractor. The CoT
    # form names the target first, so the DiT attends to object identity, then
    # relation.
    #
    # KEEP EVERY TASK'S OWN RELATION; SHARE ONE VOCABULARY. In every one of the
    # 10 tasks the BDDL target is akita_black_bowl_1 and the distractor
    # akita_black_bowl_2, so the referring expression's whole job is to separate
    # two bowls. Measured on init state 0
    # (src/kv_grounding_probe.py --list_bodies), distance ratio between the two
    # bowls under each task's OWN named anchor:
    #
    #   t0 between the plate and the ramekin  ramekin  1.20x   <-- the outlier
    #   t1 next to the ramekin                ramekin  3.61x
    #   t2 from table center                  centre   4.02x
    #   t3 on the cookie box                  cookies    inf   (stacked)
    #   t4 in the top drawer                  cabinet 13.28x   (in drawer vs on top)
    #   t5 on the ramekin                     ramekin 20.82x   (stacked)
    #   t6 next to the cookie box             cookies  3.33x
    #   t7 on the stove                       stove    2.91x   (support surface)
    #   t8 next to the plate                  plate    2.19x
    #   t9 on the wooden cabinet              cabinet 17.83x   (stacked)
    #
    # Nine of ten relations are >=2x, most of them stacking or support-surface
    # relations that are visually unambiguous. Only "between" is weak, because it
    # names a LINE rather than an anchor, and the target's position along that
    # line changes every episode. That -- not vocabulary -- is why this one task
    # fails while libero_spatial as a whole sits at ~87%.
    #
    # Which separates the two axes, and they need opposite treatment:
    #
    #  * RELATIONS stay per-task. Do NOT unify onto the plate anchor -- it is
    #    right for t0 (2.30x vs the ramekin's 1.20x) and t8, and useless
    #    elsewhere: for t1 the plate separates the bowls 1.01x, a coin flip.
    #  * VOCABULARY is global. An earlier note here argued the nine working tasks
    #    should keep "black bowl"/"ramekin", on the grounds that ~87% suite
    #    success shows the cross-attention path grounds those nouns even though
    #    qwen_color_probe's GENERATIVE path rejects them. That reasoning is
    #    sound about the nouns and wrong about the outcome, because the
    #    alternative was not "canonical everywhere" but a three-way split: the
    #    ramekin described as a shallow empty cup in one task, a small round
    #    silver container in two more, and left as "ramekin" in six. Two of those
    #    contradict each other about an object the model sees in every frame.
    #    Consistency has to win somewhere; it wins on the probe-validated
    #    wording, since that is the only one with evidence behind it.
    #
    # If the nine previously-working tasks regress on the next eval, this is the
    # change to revert -- and reverting means moving ALL ten to canonical nouns,
    # not restoring the split.
    # "between" is a SELECTOR, not a position. The scene holds TWO identical
    # black bowls: akita_black_bowl_1 is the target, akita_black_bowl_2 the
    # distractor. (Every libero_spatial task is its own BDDL scene and names
    # akita_black_bowl_1 as ITS target, so the numbering says nothing across
    # tasks -- an earlier note here called bowl_2 "also the target of the next
    # to the ramekin task", which was wrong.)
    #
    # Ground truth from the sim, one initial state (xy metres, --list_bodies):
    #   ramekin (-0.211, 0.201)   plate (0.053, 0.209)
    #   bowl_1  (-0.058, 0.210)   t=0.58 along ramekin->plate,   4mm off the line
    #   bowl_2  (-0.175, 0.323)   t=0.15,                      121mm off the line
    #
    # So the target really is ON the plate->ramekin segment -- an earlier note
    # here claimed the three objects were not collinear, which came from
    # eyeballing a screenshot and was wrong. That makes the relation
    # geometrically true but still useless as a coordinate: the target sits at
    # t=0.58 in this layout and elsewhere in others, so "between" pins the LINE
    # and nothing along it.
    #
    # Which is exactly what the policy learned. The old wording described a 1-D
    # parametric point on that axis ("in the gap between ..., closer to the
    # plate side") and it was executed literally: after the text-first fix the
    # arm stopped aiming at the midpoint and started landing on bare table on
    # the line -- right line, wrong point along it, because picking the point
    # requires seeing the bowl.
    #
    # The wording below is not guesswork -- qwen_color_probe.py was run on the
    # real env frame at 64 / 256 / 1024 vision tokens with three question sets,
    # and only one combination survived (see --probe spatial{,_visual,_plate}):
    #
    #   probe           vocabulary   anchor    @256 tok (= --vision_input_size 512)
    #   spatial         LIBERO       ramekin   FAIL "there are 0 black bowls"
    #   spatial_visual  perceptual   ramekin   FAIL relation answered backwards
    #   spatial_plate   perceptual   plate     PASS consistent + matches truth
    #
    # Two independent defects, both of which had to go:
    #
    # 1. VOCABULARY. Qwen does not see a "black bowl". At 256 tokens it replies
    #    "there are 0 black bowls ... two metallic, possibly aluminum, bowls
    #    that are silver in color", and it calls the ramekin a "small silver
    #    tray". Higher resolution makes this WORSE, since it resolves the
    #    speckled contents and reads them as metal. So: no colour words, no
    #    "ramekin" -- the containers are separated by depth, size, emptiness and
    #    contents, which is how Qwen itself described them.
    #
    #    SCOPE. This is Qwen's GENERATIVE path; the policy only ever reads
    #    cross-attention KV, and libero_spatial sits around 87%, which says that
    #    path grounds the canonical nouns well enough. So this evidence is a
    #    reason to prefer the perceptual wording, not proof the canonical wording
    #    is broken. It became decisive only because the alternative was three
    #    inconsistent vocabularies at once (see the block above).
    #
    # 2. ANCHOR, for THIS task only. "distance to the ramekin" asks the policy to
    #    measure from a landmark that sits close to the distractor bowl, and the
    #    probe answered ramekin-anchored adjacency backwards at both 64 and 256
    #    tokens. The plate is large, uniquely coloured and far from both bowls,
    #    and every plate-anchored judgement in the probe was correct from 256
    #    tokens up. Both selectors pick the target, but not equally well:
    #      "nearest the plate"    bowl_1 0.111 m vs bowl_2 0.255 m -> 2.30x
    #      "farthest from ramekin" bowl_1 0.153 m vs bowl_2 0.127 m -> 1.20x
    #    A 1.20x margin flips on a 20% distance-estimation error, which is what
    #    the probe did when anchored on the ramekin. 2.30x has room to be wrong.
    #
    #    Do NOT generalise the swap: the anchor table above measures every task,
    #    and the plate is a 1.01x coin flip for "next to the ramekin". This task
    #    is the one place the canonical anchor was the weak one.
    #
    # Still no coordinate the policy can interpolate -- "whichever of those two"
    # is a binary choice over discrete candidates, not a position. "Aim at the
    # center of the target bowl itself" pins the last step to vision. No task
    # uses cot(not_the=...) any more: negation is weak for a frozen encoder, and
    # the old "not the midpoint between the plate and the ramekin" merely
    # injected "midpoint between the plate and the ramekin". The parameter still
    # exists for callers outside libero_spatial.
    #
    # Do NOT encode the observed offset direction: placement is randomised per
    # episode, so a hard-coded "to the right" overfits one layout.
    #
    # REQUIRES --vision_input_size 512 (16x16 grid). At the default 8x8 the
    # probe FAILED its own consistency control, naming the same bowl as both
    # nearest and farthest from the plate. This wording buys nothing at 64
    # tokens; the two changes ship together or not at all.
    # All ten entries below share _spatial()'s vocabulary and differ ONLY in the
    # selector, which keeps each task's own canonical relation. The ratio after
    # each is that relation's measured separation between the two bowls.
    "pick up the black bowl between the plate and the ramekin and place it on the plate":
        _spatial(f"is nearer to {_SP_PLATE}, and the other one is not the target"),   # 2.30x
    "pick up the black bowl next to the plate and place it on the plate":
        _spatial(f"is closest to {_SP_PLATE}"),                                       # 2.19x
    "pick up the black bowl next to the ramekin and place it on the plate":
        _spatial(f"sits beside {_SP_CUP}"),                                           # 3.61x
    "pick up the black bowl on the ramekin and place it on the plate":
        _spatial(f"is resting on top of {_SP_CUP}"),                                  # 20.8x
    "pick up the black bowl from table center and place it on the plate":
        _spatial("sits in the middle of the open table, away from the edges"),        # 4.02x
    "pick up the black bowl on the cookie box and place it on the plate":
        _spatial("is resting on top of the cookie box"),                              # stacked
    "pick up the black bowl next to the cookie box and place it on the plate":
        _spatial("sits beside the cookie box, on the table surface"),                 # 3.33x
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate":
        _spatial("is inside the open top drawer of the wooden cabinet"),              # 13.3x
    "pick up the black bowl on the wooden cabinet and place it on the plate":
        _spatial("is resting on top of the wooden cabinet"),                          # 17.8x
    "pick up the black bowl on the stove and place it on the plate":
        _spatial("is resting on the stove"),                                          # 2.91x

    # ---- libero_object (20-29) — the confusable pairs only ----
    # Every libero_object scene holds ALL of these props at once and the task
    # names one to pick, so identity grounding matters more here than in
    # libero_10, where the same objects appear in pairs.
    #
    # Only the four objects whose appearance this file already asserts are
    # rewritten, reusing that wording verbatim: the two round cans (T5's rewrite
    # exists precisely because they are confusable) and the two boxes. Leaving
    # them bare here while libero_10 calls them "blue can" / "red can" would feed
    # the model two descriptions of one object.
    #
    # The other six (orange juice, ketchup, bbq sauce, milk, salad dressing,
    # chocolate pudding) are NOT rewritten: their colours are not attested
    # anywhere in this repo and guessing one wrong is worse than leaving the
    # canonical noun, which at least matches the demo language.
    "pick up the alphabet soup and place it in the basket":
        "pick up the blue can of alphabet soup and place it in the basket",
    "pick up the tomato sauce and place it in the basket":
        "pick up the red can of tomato sauce and place it in the basket",
    "pick up the cream cheese and place it in the basket":
        "pick up the silver purple cream cheese box and place it in the basket",
    "pick up the butter and place it in the basket":
        "pick up the red butter box and place it in the basket",

    # ---- libero_goal (10-19) — no rewrite ----
    # Every goal task names a single unambiguous target ("the bowl", "the plate",
    # "the wine bottle", "the cream cheese") with no same-kind distractor to
    # separate it from, so there is no referring expression to repair.
    #
    # ---- libero_10 T0 — NOT YET REWRITTEN, and the highest structural risk ----
    # "put the white mug on the left plate and put the yellow and white mug on
    # the right plate" has TWO plates told apart only by left/right -- the same
    # shape as the libero_spatial "between" failure: identical candidates
    # selected by a spatial word, where the word pins a region rather than an
    # object. Do not guess a rewrite; get ground truth first with
    #   python src/kv_grounding_probe.py --suite libero_10 --task_id 0 --list_bodies
    # and check whether left/right is stable across the 50 init states, since a
    # relation that flips between layouts cannot be encoded in the text at all.
}


def rewrite_instruction(task: str, random_augment: bool = False) -> str:
    """Return the (possibly CoT-enriched) rephrasing for `task`, or `task` unchanged.

    Args:
        task: The original task instruction string.
        random_augment: If True and a rephrasing exists, randomly choose between
            the original and rewritten version (50/50). This allows the model to
            learn BOTH phrasings during training, improving robustness at eval
            time when either form may appear. Default False (always rewrite).

    Safe to call on any string from either tasks.parquet (training) or the
    LIBERO benchmark env (RL rollout / eval) — the canonical strings match.

    NOTE ON INFERENCE-TIME CoT: because the VLM is frozen and does not
    generate text, the chain-of-thought is NOT produced at inference time —
    it is pre-written in this table and injected as the language input. The
    model is trained on these CoT strings, so at eval it simply receives the
    same CoT-enriched instruction via this function. No VLM autoregression,
    no separate "reasoning model" call, no extra latency at inference.
    """
    if not task:
        return task
    stripped = task.strip()
    rewritten = REPHRASINGS.get(stripped)
    if rewritten is None:
        return task
    if random_augment and torch.rand(1).item() > 0.5:
        return task  # 50% chance to keep original
    return rewritten


def verify_against_benchmark() -> list[str]:
    """Assert every REPHRASINGS key is a real LIBERO task string.

    Returns the list of keys that did NOT match any task across the four
    suites (should be empty). A non-empty result means a typo'd key that would
    silently no-op. Import is local so this module stays dependency-free.
    """
    from libero.libero.benchmark import get_benchmark_dict

    known: set[str] = set()
    bd = get_benchmark_dict()
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        bench = bd[suite]()
        for i in range(bench.n_tasks):
            known.add(bench.get_task(i).language.strip())

    missing = [k for k in REPHRASINGS if k.strip() not in known]
    if missing:
        print("[task_rewrites] WARNING — keys not found among LIBERO tasks "
              "(typo => silent no-op):")
        for k in missing:
            print(f"  - {k!r}")
    else:
        print(f"[task_rewrites] OK — all {len(REPHRASINGS)} rephrasing keys "
              f"match real LIBERO task strings.")
    return missing


if __name__ == "__main__":
    verify_against_benchmark()